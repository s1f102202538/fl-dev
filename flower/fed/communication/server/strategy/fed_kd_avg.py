import json
import os
from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union, override

import torch
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


class FedKDAvg(Strategy):
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

    # 増分平均用の変数
    self.cumulative_avg_logits: List[Tensor] = []  # 累積平均ロジット
    self.round_count: int = 0  # これまでに処理したラウンド数

    # 履歴強化用の変数
    self.logit_history: List[List[Tensor]] = []  # 過去のロジット履歴（時系列平滑化用）
    self.max_history_rounds: int = 5  # 保持する履歴ラウンド数

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

  def _update_incremental_average(self, new_logits: List[Tensor], server_round: int) -> List[Tensor]:
    """増分平均によるロジット更新（メモリ効率の良い方式）

    数式: avg_n = (avg_{n-1} * (n-1) + new_value) / n

    Args:
        new_logits: 現在のラウンドで集約されたロジット
        server_round: 現在のサーバーラウンド

    Returns:
        更新された累積平均ロジット
    """
    if not new_logits:
      return self.cumulative_avg_logits

    self.round_count += 1

    if not self.cumulative_avg_logits:
      # 初回ラウンドの場合
      self.cumulative_avg_logits = [logit.clone() for logit in new_logits]
      print(f"[FedKD-Incremental] Round {server_round}: Initialized cumulative average with {len(self.cumulative_avg_logits)} batches")
    else:
      # 増分平均を計算
      # avg_n = (avg_{n-1} * (n-1) + new_value) / n
      for i, new_batch_logit in enumerate(new_logits):
        if i < len(self.cumulative_avg_logits):
          prev_avg = self.cumulative_avg_logits[i]
          # 増分平均の計算
          updated_avg = (prev_avg * (self.round_count - 1) + new_batch_logit) / self.round_count
          self.cumulative_avg_logits[i] = updated_avg
        else:
          # 新しいバッチが追加された場合
          self.cumulative_avg_logits.append(new_batch_logit.clone())

      print(f"[FedKD-Incremental] Round {server_round}: Updated incremental average across {self.round_count} rounds")

    return self.cumulative_avg_logits

  def _get_enhanced_logits(self) -> List[Tensor]:
    """履歴を考慮した強化ロジット（時系列平滑化）

    時系列平滑化により、ロジットの品質と安定性を向上

    Returns:
        強化されたロジットリスト
    """
    if not self.logit_history:
      return self.avg_logits

    if len(self.logit_history) == 1:
      return self.avg_logits

    # 過去のロジットとの移動平均を計算
    current_logits = self.avg_logits
    if len(self.logit_history) >= 2:
      prev_logits = self.logit_history[-2]

      if len(current_logits) == len(prev_logits):
        # 重み付き移動平均
        alpha = 0.7  # 現在のロジットの重み
        enhanced_logits = []

        for curr_batch, prev_batch in zip(current_logits, prev_logits):
          if curr_batch.shape == prev_batch.shape:
            enhanced_batch = alpha * curr_batch + (1 - alpha) * prev_batch
            enhanced_logits.append(enhanced_batch)
          else:
            enhanced_logits.append(curr_batch)

        print(f"[FedKD-Enhanced] Applied temporal smoothing with {len(enhanced_logits)} batches (alpha={alpha})")
        return enhanced_logits

    return current_logits

  def _update_logit_history(self, new_logits: List[Tensor]) -> None:
    """ロジット履歴の更新（時系列平滑化用）

    Args:
        new_logits: 新しく集約されたロジット
    """
    if new_logits:
      # 新しいロジットを履歴に追加
      self.logit_history.append([logit.clone() for logit in new_logits])

      # 履歴の長さを制限
      if len(self.logit_history) > self.max_history_rounds:
        self.logit_history.pop(0)

      print(f"[FedKD-History] Updated logit history. Current history length: {len(self.logit_history)}")

  def _simple_average_logit_aggregation(self, logits_batch_lists: List[List[Tensor]], client_weights: List[float]) -> List[Tensor]:
    """単純な算術平均によるロジット集約（重み無し）

    Args:
        logits_batch_lists: クライアントからのロジットバッチリスト
        client_weights: クライアントの重み（未使用、互換性のために保持）

    Returns:
        算術平均で集約されたロジットリスト
    """
    if not logits_batch_lists:
      return []

    # 全クライアントで共通するバッチ数を決定
    min_batches = min(len(batches) for batches in logits_batch_lists)
    max_batches = max(len(batches) for batches in logits_batch_lists)

    if min_batches != max_batches:
      print(f"[FedKD-SA] Batch count mismatch across clients. Using {min_batches} batches (min: {min_batches}, max: {max_batches})")

    aggregated_batches = []

    # 各バッチを個別に処理（重み無し）
    for batch_idx in range(min_batches):
      batch_logits = []

      # このバッチの全クライアントロジットを収集
      for client_idx, client_batches in enumerate(logits_batch_lists):
        if batch_idx < len(client_batches):
          batch_logits.append(client_batches[batch_idx])

      if batch_logits:
        # 単純な算術平均を計算
        stacked_logits = torch.stack(batch_logits)
        averaged_logits = torch.mean(stacked_logits, dim=0)
        aggregated_batches.append(averaged_logits)

    print(f"[FedKD-SA] Successfully aggregated {len(aggregated_batches)} batches using simple average (no weights, no filtering)")
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

      # 現在のラウンドのロジット集約を実行
      current_round_logits = self._simple_average_logit_aggregation(logits_batch_lists, client_weights)

      # 増分平均で履歴を含むロジットを更新
      self.avg_logits = self._update_incremental_average(current_round_logits, server_round)

      # 履歴を更新（時系列平滑化用）
      self._update_logit_history(self.avg_logits)

      # 強化ロジットを取得（履歴を考慮した時系列平滑化）
      enhanced_logits = self._get_enhanced_logits()

      # 強化ロジットを最終的なavg_logitsとして設定
      if enhanced_logits:
        self.avg_logits = enhanced_logits

      print(
        f"[FedKD-Enhanced] Successfully aggregated {len(self.avg_logits)} batches using incremental average with temporal smoothing across {self.round_count} rounds"
      )
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
