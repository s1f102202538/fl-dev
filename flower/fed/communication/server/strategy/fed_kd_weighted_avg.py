import json
import os
from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union, override

import torch
import torch.nn.functional as F
import wandb
from fed.util.communication_cost import calculate_data_size_mb
from fed.util.model_util import base64_to_batch_list, batch_list_to_base64, create_run_dir
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, MetricsAggregationFn, Parameters, Scalar
from flwr.common.logger import log
from flwr.common.typing import UserConfig
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
from flwr.server.strategy.aggregate import weighted_loss_avg
from torch import Tensor


class FedKDWeightedAvg(Strategy):
  """Federated Knowledge Distillation with Weighted Average Logit Aggregation (FedKD-WA) strategy.

  This strategy performs knowledge distillation using weighted average aggregation of client logits.
  Key features:
  - Weighted average aggregation of client logits based on client performance
  - Quality-based filtering with batch-wise relative evaluation
  - Temperature-scaled knowledge distillation
  """

  def __init__(
    self,
    *,
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 1.0,
    min_fit_clients: int = 5,
    min_evaluate_clients: int = 5,
    min_available_clients: int = 5,
    on_fit_config_fn: Optional[Callable[[int], dict[str, Scalar]]] = None,
    on_evaluate_config_fn: Optional[Callable[[int], dict[str, Scalar]]] = None,
    accept_failures: bool = True,
    initial_parameters: Optional[Parameters] = None,
    fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
    evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
    run_config: UserConfig,
    use_wandb: bool = False,
    # Simplified logit filtering parameters
    logit_temperature: float = 3.0,  # 温度スケーリングパラメータ
    kd_temperature: float = 5.0,  # 知識蒸留用温度
    entropy_threshold: float = 0.01,  # エントロピー閾値（最小品質保証用）
    confidence_threshold: float = 0.08,  # 信頼度閾値（現実的な学習初期値）
  ) -> None:
    self.fraction_fit = fraction_fit
    self.fraction_evaluate = fraction_evaluate
    self.min_fit_clients = min_fit_clients
    self.min_evaluate_clients = min_evaluate_clients
    self.min_available_clients = min_available_clients
    self.on_fit_config_fn = on_fit_config_fn
    self.on_evaluate_config_fn = on_evaluate_config_fn
    self.accept_failures = accept_failures
    self.initial_parameters = initial_parameters
    self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
    self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
    self.avg_logits: List[Tensor] = []

    # Simplified logit filtering parameters
    self.logit_temperature = logit_temperature
    self.entropy_threshold = entropy_threshold
    self.confidence_threshold = confidence_threshold
    self.kd_temperature = kd_temperature

    self.save_path, self.run_dir = create_run_dir(run_config)
    self.use_wandb = use_wandb

    # Initialise W&B if set
    if use_wandb:
      self._init_wandb_project()

    # A dictionary to store results as they come
    self.results: Dict = {}

    # 通信コスト追跡用の変数
    self.communication_costs: Dict[str, List[float]] = {
      "server_to_client_logits_mb": [],  # サーバーからクライアントへのロジット送信コスト
      "client_to_server_logits_mb": [],  # クライアントからサーバへのロジット送信コスト
      "total_round_mb": [],  # ラウンドごとの総通信コスト
    }

  def _init_wandb_project(self) -> None:
    """Initialize W&B project."""
    wandb_project_name = os.getenv("WANDB_PROJECT_NAME", "federated-learning-default")
    wandb.init(project=wandb_project_name, name=f"{str(self.run_dir)}-ServerApp-FedKD")

  def _store_results(self, tag: str, results_dict: Dict) -> None:
    """Store results in dictionary, then save as JSON."""
    # Update results dict
    if tag in self.results:
      self.results[tag].append(results_dict)
    else:
      self.results[tag] = [results_dict]

    # Save results to disk
    with open(f"{self.save_path}/results.json", "w", encoding="utf-8") as fp:
      json.dump(self.results, fp)

  def store_results_and_log(self, server_round: int, tag: str, results_dict: Dict) -> None:
    """A helper method that stores results and logs them to W&B if enabled."""
    # Store results
    self._store_results(
      tag=tag,
      results_dict={"round": server_round, **results_dict},
    )

    if self.use_wandb:
      # Log metrics to W&B
      wandb.log(results_dict, step=server_round)

  def _evaluate_logit_quality(self, logits: Tensor) -> Dict[str, float]:
    """ロジットの品質を評価する（Non-IID環境対応版）

    Args:
        logits: 評価対象のロジットテンソル

    Returns:
        品質メトリクスの辞書
    """
    with torch.no_grad():
      # 数値安定性のためのクリッピング
      logits_clipped = torch.clamp(logits, min=-20, max=20)

      # 温度スケーリングありとなしの確率を計算
      probs_raw = F.softmax(logits_clipped, dim=1)
      probs_temp = F.softmax(logits_clipped / self.logit_temperature, dim=1)

      # 数値安定性のためeps追加
      eps = 1e-8
      probs_raw = torch.clamp(probs_raw, min=eps, max=1.0 - eps)
      probs_temp = torch.clamp(probs_temp, min=eps, max=1.0 - eps)

      # 基本品質指標
      entropy = -torch.sum(probs_raw * torch.log(probs_raw), dim=1).mean().item()
      max_prob = probs_raw.max(dim=1)[0].mean().item()
      logit_variance = logits_clipped.var(dim=1).mean().item()

      # Non-IID環境向けの追加指標
      # 1. クラス分布の均一性（Jensen-Shannon divergence）
      uniform_dist = torch.ones_like(probs_raw[0]) / probs_raw.shape[1]
      js_divergence = (
        0.5
        * (F.kl_div(torch.log(probs_raw.mean(0)), uniform_dist, reduction="sum") + F.kl_div(torch.log(uniform_dist), probs_raw.mean(0), reduction="sum")).item()
      )

      # 2. 予測の一貫性（batch内の標準偏差）
      prediction_consistency = 1.0 - probs_raw.std(dim=0).mean().item()

      # 3. 温度調整後のエントロピー
      temp_entropy = -torch.sum(probs_temp * torch.log(probs_temp), dim=1).mean().item()

      # 4. 信頼度スコア（エントロピーと最大確率の組み合わせ）
      confidence_score = max_prob * (1.0 / (1.0 + entropy))

      # 5. Non-IID指標（クラス偏り検出）
      class_distribution = probs_raw.mean(0)
      non_iid_score = torch.std(class_distribution).item()

      return {
        "entropy": entropy,
        "max_prob": max_prob,
        "logit_variance": logit_variance,
        "temp_entropy": temp_entropy,
        "confidence_score": confidence_score,
        "js_divergence": js_divergence,
        "prediction_consistency": prediction_consistency,
        "non_iid_score": non_iid_score,
        "concentration": 1.0 / (entropy + eps),
      }

  def _should_filter_logit(self, quality: Dict[str, float]) -> Tuple[bool, str]:
    """ロジットをフィルタリングすべきかを判定（無効化済み）

    Args:
        quality: ロジットの品質メトリクス

    Returns:
        常に(False, "filtering_disabled")を返す
    """
    # 品質フィルタリングを無効化 - 全てのロジットを受け入れる
    return False, "filtering_disabled"

  def _relative_quality_filter(self, batch_qualities: List[Dict[str, float]], target_keep_ratio: float = 0.7) -> List[bool]:
    """相対的品質に基づくフィルタリング（簡素化版）

    Args:
        batch_qualities: バッチ内の全ロジット品質リスト
        target_keep_ratio: 保持したいロジットの割合（0.0-1.0）

    Returns:
        各ロジットを保持するかのboolean リスト
    """
    if not batch_qualities:
      return []

    # 信頼度スコアでソート
    quality_scores = [(i, q["confidence_score"]) for i, q in enumerate(batch_qualities)]
    quality_scores.sort(key=lambda x: x[1], reverse=True)

    # 保持する数を計算
    num_to_keep = max(1, int(len(batch_qualities) * target_keep_ratio))

    # 上位を保持
    keep_indices = set(idx for idx, _ in quality_scores[:num_to_keep])

    return [i in keep_indices for i in range(len(batch_qualities))]

  def _weighted_average_logit_aggregation(self, logits_batch_lists: List[List[Tensor]], client_weights: List[float]) -> List[Tensor]:
    """データ対応関係を保持する加重平均ロジット集約

    重要: 公開データセットとの対応関係を維持するため、バッチごとに
    品質ベース選択 + 加重平均を行い、各バッチに対して必ず1つの集約ロジットを生成

    集約方式:
    1. バッチごとの品質評価によるクライアント選択
    2. 選択されたクライアントロジットの加重平均計算
    3. クライアント重みを考慮した最終集約

    Args:
        logits_batch_lists: クライアントからのロジットバッチリスト
        client_weights: クライアントの重み（加重平均用）

    Returns:
        公開データと1:1対応する加重平均集約済みロジットリスト
    """
    if not logits_batch_lists:
      return []

    # 全クライアントで共通するバッチ数を決定
    min_batches = min(len(batches) for batches in logits_batch_lists)
    max_batches = max(len(batches) for batches in logits_batch_lists)

    if min_batches != max_batches:
      print(f"[FedKD-WA] Batch count mismatch across clients. Using {min_batches} batches (min: {min_batches}, max: {max_batches})")

    # 重みを正規化
    total_weight = sum(client_weights)
    normalized_weights = [w / total_weight for w in client_weights]

    aggregated_batches = []
    batch_quality_metrics = []
    total_filtered = 0
    total_evaluated = 0

    # 各バッチを個別に処理（データ対応関係保持）
    for batch_idx in range(min_batches):
      batch_logits_candidates = []  # (client_idx, logits, quality, weight)

      # Step 1: このバッチの全クライアントロジットを評価
      for client_idx, client_batches in enumerate(logits_batch_lists):
        if batch_idx < len(client_batches):
          logits = client_batches[batch_idx]
          quality = self._evaluate_logit_quality(logits)
          batch_logits_candidates.append((client_idx, logits, quality, normalized_weights[client_idx]))
          total_evaluated += 1

      if not batch_logits_candidates:
        # このバッチにはロジットがない場合のフォールバック
        print(f"[FedKD-WA] Warning: No logits for batch {batch_idx}")
        continue

      # Step 2: このバッチ内での相対品質評価
      def composite_quality_score(quality_metrics):
        """複合品質スコア（高いほど良い）"""
        confidence = quality_metrics["confidence_score"]
        entropy_penalty = 1.0 / (1.0 + quality_metrics["entropy"])
        consistency = quality_metrics["prediction_consistency"]
        return 0.4 * confidence + 0.3 * entropy_penalty + 0.3 * consistency

      # 品質順でソート（降順：高品質が先頭）
      batch_logits_candidates.sort(key=lambda x: composite_quality_score(x[2]), reverse=True)

      # Step 3: 固定保持率に基づいて選択
      num_candidates = len(batch_logits_candidates)
      keep_ratio = 0.7  # 固定保持率
      num_to_keep = max(1, int(num_candidates * keep_ratio))  # 最低1つは保持

      selected_candidates = batch_logits_candidates[:num_to_keep]
      filtered_count = num_candidates - num_to_keep
      total_filtered += filtered_count

      # Step 4: 選択されたロジットで重み付き集約
      if len(selected_candidates) == 1:
        # 1つのロジットのみ: そのまま使用
        _, logits, quality, _ = selected_candidates[0]
        aggregated_batches.append(logits)
        batch_quality_metrics.append(quality)
      else:
        # 複数ロジット: クライアント重みベースの加重平均集約
        batch_logits = [candidate[1] for candidate in selected_candidates]
        batch_weights = [candidate[3] for candidate in selected_candidates]

        # 加重平均用の重み正規化
        total_batch_weight = sum(batch_weights)
        if total_batch_weight > 0:
          batch_weights = [w / total_batch_weight for w in batch_weights]

        # 加重平均による集約（Weighted Average Aggregation）
        stacked_logits = torch.stack(batch_logits)
        weight_tensor = torch.tensor(batch_weights, device=stacked_logits.device).view(-1, 1, 1)
        weighted_logits = (stacked_logits * weight_tensor).sum(dim=0)

        # 集約品質を評価
        aggregated_quality = self._evaluate_logit_quality(weighted_logits)
        aggregated_batches.append(weighted_logits)
        batch_quality_metrics.append(aggregated_quality)

      # バッチ単位でのフィルタリング状況をログ出力（詳細モード）
      if batch_idx % 50 == 0 or filtered_count > 0:
        selected_clients = [candidate[0] for candidate in selected_candidates]
        filtered_clients = [candidate[0] for candidate in batch_logits_candidates[num_to_keep:]]
        if filtered_count > 0:
          print(f"[FedKD-WA] Batch {batch_idx}: kept clients {selected_clients}, filtered clients {filtered_clients}")

    # Step 5: 全体統計とログ出力
    if batch_quality_metrics and total_evaluated > 0:
      overall_quality = {
        "confidence_score": sum(q["confidence_score"] for q in batch_quality_metrics) / len(batch_quality_metrics),
        "entropy": sum(q["entropy"] for q in batch_quality_metrics) / len(batch_quality_metrics),
        "non_iid_score": sum(q.get("non_iid_score", 0) for q in batch_quality_metrics) / len(batch_quality_metrics),
        "prediction_consistency": sum(q["prediction_consistency"] for q in batch_quality_metrics) / len(batch_quality_metrics),
      }

      actual_filter_rate = total_filtered / total_evaluated * 100 if total_evaluated > 0 else 0.0

      print("[FedKD-WA] === Weighted Average Aggregation Report ===")
      print(f"  🎯 Filtering Rate: {actual_filter_rate:.1f}%")
      print(f"  🔢 Filtered: {total_filtered}/{total_evaluated} client logits")
      print(f"  📦 Output Batches: {len(aggregated_batches)} (= input {min_batches})")
      print("  🔗 Data Correspondence: MAINTAINED (1:1 mapping)")
      print(f"  📋 Avg Quality - Confidence: {overall_quality['confidence_score']:.4f}, Entropy: {overall_quality['entropy']:.4f}")
      print("  ⚖️  Aggregation Method: Weighted Average (client performance based)")
      print("  ============================================")

      # データ対応関係の確認
      if len(aggregated_batches) == min_batches:
        print("  ✅ Perfect data correspondence maintained")
      else:
        print(f"  ⚠️  Data correspondence issue: {len(aggregated_batches)} ≠ {min_batches}")

    return aggregated_batches

  @override
  def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
    """Initialize the (global) model parameters.

    Parameters
    ----------
    client_manager : ClientManager
        The client manager which holds all currently connected clients.

    Returns
    -------
    parameters : Optional[Parameters]
        If parameters are returned, then the server will treat these as the
        initial global model parameters.
    """

    # サーバはモデルを持たないためモデルパラメータは None を返す
    return None

  @override
  def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy, FitIns]]:
    """Configure the next round of training with enhanced logits and communication cost measurement."""

    config = {}
    # 現在のラウンド情報を追加
    config["current_round"] = server_round

    # 集約されたロジット（簡素化版）
    enhanced_logits = self.avg_logits

    # サーバーからクライアントへの通信コスト測定
    server_to_client_mb = 0.0

    # 前回のラウンドで集約されたロジットがある場合のみ追加
    if enhanced_logits:
      logits_data = batch_list_to_base64(enhanced_logits)
      config["avg_logits"] = logits_data
      # ロジットデータのサイズを測定
      server_to_client_mb = calculate_data_size_mb(logits_data)
      # 固定温度をクライアントに送信
      config["temperature"] = self.kd_temperature
      config["logit_temperature"] = self.logit_temperature
      print(
        f"[FedKD] Sending {len(enhanced_logits)} enhanced logit batches (KD temp: {self.kd_temperature:.3f}, logit temp: {self.logit_temperature:.3f}, size: {server_to_client_mb:.4f} MB)"
      )
    else:
      print("[FedKD] No logits available for this round")

    # 有効になっているクライアントの取得
    sample_size = int(self.fraction_fit * client_manager.num_available())
    clients = client_manager.sample(num_clients=sample_size, min_num_clients=self.min_fit_clients)

    # 実際の通信コストはクライアント数を考慮
    total_server_to_client_mb = server_to_client_mb * len(clients)
    self.communication_costs["server_to_client_logits_mb"].append(total_server_to_client_mb)

    print(f"[FedKD] Total server->client communication: {total_server_to_client_mb:.4f} MB ({len(clients)} clients)")

    fit_ins = FitIns(parameters, config)
    return [(client, fit_ins) for client in clients]

  @override
  def aggregate_fit(
    self,
    server_round: int,
    results: List[Tuple[ClientProxy, FitRes]],
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
  ) -> Tuple[Optional[Parameters], dict[str, Scalar]]:
    """Aggregate training results with enhanced logit processing and communication cost measurement."""

    logits_batch_lists = []
    client_weights = []

    # 通信コスト測定
    total_logits_mb = 0.0

    for _, fit_res in results:
      # ロジットサイズ測定
      if fit_res.metrics and "logits" in fit_res.metrics:
        logits_data = str(fit_res.metrics["logits"])
        logits_size_mb = calculate_data_size_mb(logits_data)
        total_logits_mb += logits_size_mb

      if "logits" in fit_res.metrics:
        # バッチリスト形式でロジットを取得
        logits_batch_list = base64_to_batch_list(str(fit_res.metrics["logits"]))

        logits_batch_lists.append(logits_batch_list)
        # クライアントの重み（データサイズベース）
        client_weights.append(float(fit_res.num_examples))

    if logits_batch_lists and client_weights:
      print(f"[FedKD] Aggregating logits from {len(logits_batch_lists)} clients")

      # 重み付きロジット集約を実行
      self.avg_logits = self._weighted_average_logit_aggregation(logits_batch_lists, client_weights)

      print(f"[FedKD-WA] Successfully aggregated {len(self.avg_logits)} batches using weighted average")
    else:
      print("[FedKD] No valid logits received from clients")

    # 通信コストを記録
    self.communication_costs["client_to_server_logits_mb"].append(total_logits_mb)

    # ラウンドの総通信コストを計算
    server_to_client_logits_mb = self.communication_costs["server_to_client_logits_mb"][-1] if self.communication_costs["server_to_client_logits_mb"] else 0.0
    total_round_mb = server_to_client_logits_mb + total_logits_mb
    self.communication_costs["total_round_mb"].append(total_round_mb)

    print(
      f"[FedKD] Round {server_round}: Server->Client: {server_to_client_logits_mb:.4f} MB, Client->Server logits: {total_logits_mb:.4f} MB, total: {total_round_mb:.4f} MB"
    )

    # メトリクスの集約
    aggregated_metrics = {}
    if self.fit_metrics_aggregation_fn:
      fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
      aggregated_metrics = self.fit_metrics_aggregation_fn(fit_metrics)

    # 現在の温度をメトリクスに追加
    aggregated_metrics["current_kd_temperature"] = self.kd_temperature
    aggregated_metrics["current_logit_temperature"] = self.logit_temperature
    if self.avg_logits:
      aggregated_metrics["num_aggregated_batches"] = len(self.avg_logits)

    # 通信コストをメトリクスに追加
    aggregated_metrics["comm_cost_server_to_client_mb"] = server_to_client_logits_mb
    aggregated_metrics["comm_cost_client_to_server_logits_mb"] = total_logits_mb
    aggregated_metrics["comm_cost_total_round_mb"] = total_round_mb
    aggregated_metrics["comm_cost_cumulative_mb"] = sum(self.communication_costs["total_round_mb"])

    # 通信コストメトリクスをW&Bにログ
    communication_metrics = {
      "comm_cost_server_to_client_mb": server_to_client_logits_mb,
      "comm_cost_client_to_server_logits_mb": total_logits_mb,
      "comm_cost_total_round_mb": total_round_mb,
      "comm_cost_cumulative_mb": sum(self.communication_costs["total_round_mb"]),
      "current_kd_temperature": self.kd_temperature,
      "current_logit_temperature": self.logit_temperature,
    }

    # 品質メトリクスは削除済み（簡素化のため）

    if self.avg_logits:
      communication_metrics["num_aggregated_batches"] = len(self.avg_logits)

    self.store_results_and_log(server_round=server_round, tag="communication_costs", results_dict=communication_metrics)

    return None, aggregated_metrics

  @override
  def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy, EvaluateIns]]:
    """Configure the next round of evaluation.

    Parameters
    ----------
    server_round : int
        The current round of federated learning.
    parameters : Parameters
        The current (global) model parameters.
    client_manager : ClientManager
        The client manager which holds all currently connected clients.

    Returns
    -------
    evaluate_configuration : List[Tuple[ClientProxy, EvaluateIns]]
        A list of tuples. Each tuple in the list identifies a `ClientProxy` and the
        `EvaluateIns` for this particular `ClientProxy`. If a particular
        `ClientProxy` is not included in this list, it means that this
        `ClientProxy` will not participate in the next round of federated
        evaluation.
    """

    # 評価用の設定を作成
    config = {}

    # 現在のラウンド情報を追加
    config["current_round"] = server_round

    # 前回のラウンドで集約されたロジットがある場合のみ追加
    if self.avg_logits:
      config["avg_logits"] = batch_list_to_base64(self.avg_logits)
    # 初回ラウンドではロジットが存在しないため、avg_logitsキーを含めない
    evaluate_ins = EvaluateIns(parameters, config)

    # 評価に参加するクライアントをサンプリング
    sample_size = int(self.fraction_evaluate * client_manager.num_available())
    clients = client_manager.sample(num_clients=sample_size, min_num_clients=self.min_evaluate_clients)

    # Return client/config pairs
    return [(client, evaluate_ins) for client in clients]

  @override
  def aggregate_evaluate(
    self,
    server_round: int,
    results: List[Tuple[ClientProxy, EvaluateRes]],
    failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
  ) -> Tuple[Optional[float], Dict[str, Scalar]]:
    """Aggregate evaluation results.

    Parameters
    ----------
    server_round : int
        The current round of federated learning.
    results : List[Tuple[ClientProxy, FitRes]]
        Successful updates from the
        previously selected and configured clients. Each pair of
        `(ClientProxy, FitRes` constitutes a successful update from one of the
        previously selected clients. Not that not all previously selected
        clients are necessarily included in this list: a client might drop out
        and not submit a result. For each client that did not submit an update,
        there should be an `Exception` in `failures`.
    failures : List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]
        Exceptions that occurred while the server was waiting for client updates.

    Returns
    -------
    aggregation_result : Tuple[Optional[float], Dict[str, Scalar]]
        The aggregated evaluation result. Aggregation typically uses some variant
        of a weighted average.
    """

    if not results:
      return None, {}
    # Do not aggregate if there are failures and failures are not accepted
    if not self.accept_failures and failures:
      return None, {}

    # Aggregate loss
    loss_aggregated = weighted_loss_avg([(evaluate_res.num_examples, evaluate_res.loss) for _, evaluate_res in results])

    # Aggregate custom metrics if aggregation fn was provided
    metrics_aggregated = {}
    if self.evaluate_metrics_aggregation_fn:
      eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
      metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
    elif server_round == 1:  # Only log this warning once
      log(WARNING, "No evaluate_metrics_aggregation_fn provided")

    # 精度情報のログ出力
    if "accuracy" in metrics_aggregated:
      accuracy = metrics_aggregated["accuracy"]
      print(f"[FedKD] Round {server_round} - Accuracy: {accuracy:.4f}, Loss: {loss_aggregated:.4f}")

    # Store and log FedKD evaluation results
    self.store_results_and_log(
      server_round=server_round,
      tag="federated_evaluate",
      results_dict={"federated_evaluate_loss": loss_aggregated, **metrics_aggregated},
    )

    return loss_aggregated, metrics_aggregated

  @override
  def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
    """Evaluate the current model parameters.

    FedKD uses logit-based knowledge distillation instead of parameter aggregation.
    Server-side centralized evaluation is not applicable for this strategy.

    Parameters
    ----------
    server_round : int
        The current round of federated learning.
    parameters: Parameters
        The current (global) model parameters (unused in FedKD).

    Returns
    -------
    evaluation_result : Optional[Tuple[float, Dict[str, Scalar]]]
        Always returns None as FedKD does not perform centralized evaluation.
    """
    # FedKDはロジットベースの知識蒸留を使用するため、
    # サーバー側でのパラメータベース評価は行わない
    return None
