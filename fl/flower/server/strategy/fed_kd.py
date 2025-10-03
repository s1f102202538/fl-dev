import json
from logging import INFO, WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union, override

import torch
import torch.nn.functional as F
import wandb
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, MetricsAggregationFn, Parameters, Scalar
from flwr.common.logger import log
from flwr.common.typing import UserConfig
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
from flwr.server.strategy.aggregate import weighted_loss_avg
from torch import Tensor

from flower.common.util.util import base64_to_batch_list, batch_list_to_base64, create_run_dir

PROJECT_NAME = "fl-dev-cifer-10"


class FedKD(Strategy):
  """Federated Knowledge Distillation (FedKD) strategy."""

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
    inplace: bool = True,
    run_config: UserConfig,
    use_wandb: bool = False,
    # 新しいパラメータ
    logit_temperature: float = 3.0,
    kd_temperature: float = 3.0,
    enable_adaptive_temperature: bool = True,
    entropy_threshold: float = 0.5,
    max_history_rounds: int = 3,
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
    self.inplace = inplace
    self.avg_logits: List[Tensor] = []

    self.logit_temperature = logit_temperature
    self.enable_adaptive_temperature = enable_adaptive_temperature
    self.entropy_threshold = entropy_threshold
    self.max_history_rounds = max_history_rounds

    # ロジット履歴とメトリクス
    self.logit_history: List[List[Tensor]] = []
    self.kd_temperature = kd_temperature
    self.round_metrics = []

    self.save_path, self.run_dir = create_run_dir(run_config)
    self.use_wandb = use_wandb

    # Initialise W&B if set
    if use_wandb:
      self._init_wandb_project()

    # Keep track of best acc
    self.best_acc_so_far = 0.0

    # A dictionary to store results as they come
    self.results: Dict = {}

  def _init_wandb_project(self) -> None:
    """Initialize W&B project."""
    wandb.init(project=PROJECT_NAME, name=f"{str(self.run_dir)}-ServerApp-FedKD")

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
    """ロジットの品質を評価する"""
    with torch.no_grad():
      # ソフトマックス確率を計算
      probs = F.softmax(logits / self.logit_temperature, dim=1)

      # エントロピーを計算（不確実性の指標）
      entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean().item()

      # 最大確率（信頼度の指標）
      max_prob = probs.max(dim=1)[0].mean().item()

      # ロジットの分散（多様性の指標）
      logit_variance = logits.var(dim=1).mean().item()

      # 温度調整後のエントロピー
      temp_entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean().item()

      return {
        "entropy": entropy,
        "max_prob": max_prob,
        "logit_variance": logit_variance,
        "temp_entropy": temp_entropy,
      }

  def _weighted_logit_aggregation(self, logits_batch_lists: List[List[Tensor]], client_weights: List[float]) -> List[Tensor]:
    """重み付きロジット集約"""
    if not logits_batch_lists:
      return []

    # 全クライアントで共通するバッチ数を決定
    min_batches = min(len(batches) for batches in logits_batch_lists)
    max_batches = max(len(batches) for batches in logits_batch_lists)

    if min_batches != max_batches:
      print(f"[FedKD] Batch count mismatch across clients. Using {min_batches} batches (min: {min_batches}, max: {max_batches})")

    # 重みを正規化
    total_weight = sum(client_weights)
    normalized_weights = [w / total_weight for w in client_weights]

    aggregated_batches = []
    batch_quality_metrics = []

    for batch_idx in range(min_batches):
      batch_logits = []
      batch_weights = []

      for client_idx, client_batches in enumerate(logits_batch_lists):
        if batch_idx < len(client_batches):
          logits = client_batches[batch_idx]

          # ロジットの品質をチェック
          quality = self._evaluate_logit_quality(logits)

          # 低品質のロジットを除外（エントロピーが低すぎる場合）
          if quality["entropy"] > self.entropy_threshold:
            batch_logits.append(logits)
            batch_weights.append(normalized_weights[client_idx])
          else:
            print(f"[FedKD] Skipping low-quality logits from client {client_idx}, batch {batch_idx} (entropy: {quality['entropy']:.4f})")

      if batch_logits:
        # 重み付き平均を計算
        if len(batch_weights) > 1:
          # 重みを再正規化
          total_batch_weight = sum(batch_weights)
          batch_weights = [w / total_batch_weight for w in batch_weights]

          # 重み付き集約（torch.stackを使用）
          stacked_logits = torch.stack(batch_logits)
          weight_tensor = torch.tensor(batch_weights, device=stacked_logits.device).view(-1, 1, 1)
          weighted_logits = (stacked_logits * weight_tensor).sum(dim=0)
        else:
          weighted_logits = batch_logits[0]

        # 集約されたロジットの品質を評価
        aggregated_quality = self._evaluate_logit_quality(weighted_logits)
        batch_quality_metrics.append(aggregated_quality)

        aggregated_batches.append(weighted_logits)

    # バッチ品質メトリクスをログ出力
    if batch_quality_metrics:
      avg_entropy = sum(m["entropy"] for m in batch_quality_metrics) / len(batch_quality_metrics)
      avg_max_prob = sum(m["max_prob"] for m in batch_quality_metrics) / len(batch_quality_metrics)
      print(f"[FedKD] Aggregated logits quality - Avg entropy: {avg_entropy:.4f}, Avg max prob: {avg_max_prob:.4f}")

    return aggregated_batches

  def _manage_logit_history(self, new_logits: List[Tensor]) -> None:
    """ロジット履歴の管理"""
    if new_logits:
      self.logit_history.append(new_logits.copy())

      # 履歴サイズを制限
      if len(self.logit_history) > self.max_history_rounds:
        self.logit_history.pop(0)

  def _get_enhanced_logits(self) -> List[Tensor]:
    """履歴を考慮した強化ロジット"""
    if not self.logit_history:
      return self.avg_logits

    if len(self.logit_history) == 1:
      return self.avg_logits

    # 過去のロジットとの移動平均を計算
    current_logits = self.avg_logits
    if len(self.logit_history) >= 2:
      prev_logits = self.logit_history[-2]

      if len(current_logits) == len(prev_logits):
        # 重み付き移動平均（現在のロジットを重視）
        alpha = 0.7  # 現在のロジットの重み
        enhanced_logits = []

        for curr_batch, prev_batch in zip(current_logits, prev_logits):
          if curr_batch.shape == prev_batch.shape:
            enhanced_batch = alpha * curr_batch + (1 - alpha) * prev_batch
            enhanced_logits.append(enhanced_batch)
          else:
            enhanced_logits.append(curr_batch)

        print(f"[FedKD] Applied temporal smoothing with {len(enhanced_logits)} batches")
        return enhanced_logits

    return current_logits

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
    """Configure the next round of training with enhanced logits."""

    config = {}
    # 現在のラウンド情報を追加
    config["current_round"] = server_round

    # 強化されたロジットを取得（履歴を考慮）
    enhanced_logits = self._get_enhanced_logits()

    # 前回のラウンドで集約されたロジットがある場合のみ追加
    if enhanced_logits:
      config["avg_logits"] = batch_list_to_base64(enhanced_logits)
      # 現在の温度をクライアントに送信
      config["temperature"] = self.kd_temperature
      print(f"[FedKD] Sending {len(enhanced_logits)} enhanced logit batches to clients (temp: {self.kd_temperature:.3f})")
    else:
      print("[FedKD] No logits available for this round")

    fit_ins = FitIns(parameters, config)

    # 有効になっているクライアントの取得
    sample_size = int(self.fraction_fit * client_manager.num_available())
    clients = client_manager.sample(num_clients=sample_size, min_num_clients=self.min_fit_clients)

    return [(client, fit_ins) for client in clients]

  @override
  def aggregate_fit(
    self,
    server_round: int,
    results: List[Tuple[ClientProxy, FitRes]],
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
  ) -> Tuple[Optional[Parameters], dict[str, Scalar]]:
    """Aggregate training results with enhanced logit processing."""

    logits_batch_lists = []
    client_weights = []

    for _, fit_res in results:
      if "logits" in fit_res.metrics:
        # バッチリスト形式でロジットを取得
        logits_batch_list = base64_to_batch_list(str(fit_res.metrics["logits"]))

        # NaN/Infを含むバッチは除外
        filtered_batch_list = []
        for batch in logits_batch_list:
          if not torch.isnan(batch).any() and not torch.isinf(batch).any():
            filtered_batch_list.append(batch)
          else:
            print("[FedKD] Skipped batch with NaN/Inf in logits from a client.")

        if filtered_batch_list:
          logits_batch_lists.append(filtered_batch_list)
          # クライアントの重み（データサイズベース）
          client_weights.append(float(fit_res.num_examples))

    if logits_batch_lists and client_weights:
      print(f"[FedKD] Aggregating logits from {len(logits_batch_lists)} clients")

      # 重み付きロジット集約を実行
      self.avg_logits = self._weighted_logit_aggregation(logits_batch_lists, client_weights)

      # ロジット履歴を管理
      self._manage_logit_history(self.avg_logits)

      print(f"[FedKD] Successfully aggregated {len(self.avg_logits)} batches of logits")
    else:
      print("[FedKD] No valid logits received from clients")

    # メトリクスの集約
    aggregated_metrics = {}
    if self.fit_metrics_aggregation_fn:
      fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
      aggregated_metrics = self.fit_metrics_aggregation_fn(fit_metrics)

    # 現在の温度をメトリクスに追加
    aggregated_metrics["current_temperature"] = self.kd_temperature
    if self.avg_logits:
      aggregated_metrics["num_aggregated_batches"] = len(self.avg_logits)

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
