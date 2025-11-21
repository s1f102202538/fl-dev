import copy
import json
import os
from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union, override

import torch
import wandb
from fed.algorithms.distillation import Distillation
from fed.models.base_model import BaseModel
from fed.task.cnn_task import CNNTask
from fed.util.communication_cost import calculate_data_size_mb
from fed.util.model_util import base64_to_batch_list, batch_list_to_base64, create_run_dir
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, MetricsAggregationFn, Parameters, Scalar, ndarrays_to_parameters
from flwr.common.logger import log
from flwr.common.typing import UserConfig
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
from flwr.server.strategy.aggregate import weighted_loss_avg
from torch import Tensor
from torch.utils.data import DataLoader


class FedKDPublicDistillation(Strategy):
  """Federated Knowledge Distillation strategy with public data pre-training.

  This strategy performs knowledge distillation with a two-phase approach:

  Round 1:
  1. Server trains its model using public data (supervised learning)
  2. Server generates logits from the trained model using public data
  3. Server broadcasts these logits to clients for knowledge distillation

  Round 2+:
  1. Clients send their locally trained logits to the server
  2. Server aggregates client logits using weighted averaging
  3. Server trains its model using the aggregated logits via knowledge distillation
  4. Server generates new logits from the trained model
  5. Server broadcasts the updated logits to clients

  Key approach: Server model is pre-trained with public data in round 1,
  then continuously improved through client knowledge aggregation in subsequent rounds.
  """

  def __init__(
    self,
    *,
    server_model: BaseModel,
    public_data_loader: DataLoader,
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
    kd_temperature: float = 5.0,  # 知識蒸留用温度
    distillation_epochs: int = 5,  # 蒸留訓練のエポック数（ラウンド2以降）
    distillation_learning_rate: float = 0.001,  # 蒸留訓練の学習率（ラウンド2以降）
    public_training_epochs: int = 5,  # 公開データ訓練のエポック数（全ラウンド）
    public_training_learning_rate: float = 0.01,  # 公開データ訓練の学習率（全ラウンド）
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

    # Server-side model and training configuration
    self.server_model = server_model
    self.public_data_loader = public_data_loader
    self.distillation_epochs = distillation_epochs
    self.distillation_learning_rate = distillation_learning_rate
    self.public_training_epochs = public_training_epochs
    self.public_training_learning_rate = public_training_learning_rate
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.server_model.to(self.device)

    # Server-generated logits for distribution to clients
    self.server_generated_logits: List[Tensor] = []

    # Server model state management for persistence across rounds
    self.saved_server_model_state: Optional[Dict] = None

    # Knowledge distillation temperature
    self.kd_temperature = kd_temperature

    # Flag to track if round 1 training has been completed
    self.round1_training_completed = False

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
    wandb.init(project=wandb_project_name, name=f"{str(self.run_dir)}-ServerApp-FedKD-PublicDistillation")

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

  def _train_server_model_on_public_data(self, server_round: int) -> None:
    """Train server model on public data using supervised learning (Round 1 only).

    Args:
        server_round: Current federated learning round
    """
    print(f"[FedKD-PublicDist] Round {server_round}: Training server model on public data (supervised learning)")

    train_loss = CNNTask.train(
      net=self.server_model,
      train_loader=self.public_data_loader,
      epochs=self.public_training_epochs,
      lr=self.public_training_learning_rate,
      device=self.device,
    )

    print(
      f"[FedKD-PublicDist] Round {server_round}: Server model training on public data completed "
      f"(epochs: {self.public_training_epochs}, lr: {self.public_training_learning_rate:.4f}, loss: {train_loss:.4f})"
    )

    # Mark round 1 training as completed
    self.round1_training_completed = True

  def _aggregate_client_logits(self, logits_batch_lists: List[List[Tensor]], client_weights: List[float]) -> List[Tensor]:
    """Aggregate client logits using simple averaging.

    Args:
        logits_batch_lists: List of logit batch lists from each client
        client_weights: Weights for each client (ignored, kept for compatibility)

    Returns:
        List of aggregated logit batches
    """
    if not logits_batch_lists:
      return []

    print("[FedKD-PublicDist] Using simple averaging for logit aggregation")

    # Get the number of batches (assuming all clients have the same number of batches)
    num_batches = len(logits_batch_lists[0])

    # Check that all clients have the same number of batches
    for i, client_logits in enumerate(logits_batch_lists):
      if len(client_logits) != num_batches:
        print(f"[FedKD-PublicDist] WARNING: Client {i} has {len(client_logits)} batches, expected {num_batches}")

    aggregated_logits = []

    # Aggregate each batch position across all clients using simple averaging
    for batch_idx in range(num_batches):
      # Collect logits from all clients for this batch position
      batch_logits_list = []

      for client_idx, client_logits in enumerate(logits_batch_lists):
        if batch_idx < len(client_logits):  # Safety check
          batch_logits_list.append(client_logits[batch_idx])

      if batch_logits_list:
        # Simple arithmetic mean of logits for this batch
        stacked_logits = torch.stack(batch_logits_list)
        aggregated_batch = torch.mean(stacked_logits, dim=0)
        aggregated_logits.append(aggregated_batch)

    print(f"[FedKD-PublicDist] Aggregated {len(aggregated_logits)} batches using simple averaging")
    return aggregated_logits

  def _train_server_model_with_aggregated_logits(self, logits_batch_lists: List[List[Tensor]], client_weights: List[float], server_round: int) -> None:
    """Train server model using aggregated client logits via knowledge distillation.

    Args:
        logits_batch_lists: List of logit batch lists from each client
        client_weights: Weights for each client
        server_round: Current federated learning round
    """
    if not logits_batch_lists:
      print(f"[FedKD-PublicDist] Round {server_round}: No client logits available for server training")
      return

    total_clients = len(logits_batch_lists)
    print(f"[FedKD-PublicDist] Round {server_round}: Aggregating logits from {total_clients} clients for server training")

    # Aggregate client logits using weighted averaging
    aggregated_logits = self._aggregate_client_logits(logits_batch_lists, client_weights)

    if not aggregated_logits:
      print(f"[FedKD-PublicDist] Round {server_round}: No aggregated logits available for server training")
      return

    print(f"[FedKD-PublicDist] Round {server_round}: Training server model with {len(aggregated_logits)} aggregated logit batches")

    # Create distillation trainer with aggregated logits
    distillation = Distillation(
      studentModel=self.server_model,
      public_data=self.public_data_loader,
      soft_targets=aggregated_logits,
    )

    # Train server model using knowledge distillation with aggregated logits
    self.server_model = distillation.train_knowledge_distillation(
      epochs=self.distillation_epochs,
      learning_rate=self.distillation_learning_rate,
      T=self.kd_temperature,
      alpha=0.3,  # KL distillation loss weight
      beta=0.7,  # CE loss weight
      device=self.device,
    )

    print(
      f"[FedKD-PublicDist] Round {server_round}: Server model distillation training completed with aggregated logits "
      f"({self.distillation_epochs} epochs, lr: {self.distillation_learning_rate:.4f})"
    )

  def _generate_server_logits(self, server_round: int) -> List[Tensor]:
    """Generate logits using trained server model on public data.

    Args:
        server_round: Current federated learning round

    Returns:
        List of logits generated by server model
    """
    print(f"[FedKD-PublicDist] Round {server_round}: Generating logits from trained server model")

    # Generate logits using trained server model
    server_logits = CNNTask.inference(self.server_model, self.public_data_loader, device=self.device)

    print(f"[FedKD-PublicDist] Round {server_round}: Generated {len(server_logits)} server logit batches")

    return server_logits

  def _save_server_model_state(self, server_round: int) -> None:
    """Save current server model state for next round persistence.

    Args:
        server_round: Current federated learning round
    """
    print(f"[FedKD-PublicDist] Round {server_round}: Saving server model state for next round")

    # Deep copy of the current state to avoid reference issues
    self.saved_server_model_state = copy.deepcopy(self.server_model.state_dict())

    print(f"[FedKD-PublicDist] Round {server_round}: Server model state saved successfully")

  def _restore_server_model_state(self, server_round: int) -> None:
    """Restore server model state from previous round if available.

    Args:
        server_round: Current federated learning round
    """
    if server_round == 1:
      print(f"[FedKD-PublicDist] Round {server_round}: First round - using initial server model state")
      return

    if self.saved_server_model_state is not None:
      print(f"[FedKD-PublicDist] Round {server_round}: Restoring server model state from previous round")
      self.server_model.load_state_dict(self.saved_server_model_state)
      print(f"[FedKD-PublicDist] Round {server_round}: Server model state restored successfully")
    else:
      print(f"[FedKD-PublicDist] Round {server_round}: WARNING - No saved state found, using current model state")

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
    """Configure the next round of training with server-generated logits and communication cost measurement."""

    # Round 1: Train server model on public data and generate initial logits
    if server_round == 1 and not self.round1_training_completed:
      print(f"[FedKD-PublicDist] Round {server_round}: Performing initial server training on public data")
      self._train_server_model_on_public_data(server_round)
      self.server_generated_logits = self._generate_server_logits(server_round)
      self._save_server_model_state(server_round)

    config = {}
    # 現在のラウンド情報を追加
    config["current_round"] = server_round

    # サーバーからクライアントへの通信コスト測定
    server_to_client_mb = 0.0

    # サーバーで生成されたロジットがある場合のみ追加
    if self.server_generated_logits:
      logits_data = batch_list_to_base64(self.server_generated_logits)
      config["avg_logits"] = logits_data
      # ロジットデータのサイズを測定
      server_to_client_mb = calculate_data_size_mb(logits_data)
      # 現在の温度をクライアントに送信
      config["temperature"] = self.kd_temperature
      print(
        f"[FedKD-PublicDist] Round {server_round}: Sending {len(self.server_generated_logits)} server-generated logit batches to clients "
        f"(temp: {self.kd_temperature:.3f}, size: {server_to_client_mb:.4f} MB)"
      )
    else:
      print(f"[FedKD-PublicDist] Round {server_round}: WARNING - No server logits available")

    # 有効になっているクライアントの取得
    sample_size = int(self.fraction_fit * client_manager.num_available())
    clients = client_manager.sample(num_clients=sample_size, min_num_clients=self.min_fit_clients)

    # 実際の通信コストはクライアント数を考慮
    total_server_to_client_mb = server_to_client_mb * len(clients)
    self.communication_costs["server_to_client_logits_mb"].append(total_server_to_client_mb)

    print(f"[FedKD-PublicDist] Round {server_round}: Total server->client communication: {total_server_to_client_mb:.4f} MB ({len(clients)} clients)")

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

    # Restore server model state from previous round before training (for rounds 2+)
    if server_round > 1:
      self._restore_server_model_state(server_round)

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

    # All rounds: Train server model with aggregated client logits, then train on public data
    if logits_batch_lists and client_weights:
      print(f"[FedKD-PublicDist] Round {server_round}: Processing logits from {len(logits_batch_lists)} clients for aggregated training")

      # Step 1: Train server model with aggregated client logits (knowledge distillation)
      self._train_server_model_with_aggregated_logits(logits_batch_lists, client_weights, server_round)

      # Step 2: Further train the model on public data (supervised learning)
      print(f"[FedKD-PublicDist] Round {server_round}: Further training server model on public data after distillation")
      train_loss = CNNTask.train(
        net=self.server_model,
        train_loader=self.public_data_loader,
        epochs=self.public_training_epochs,
        lr=self.public_training_learning_rate,
        device=self.device,
      )
      print(
        f"[FedKD-PublicDist] Round {server_round}: Public data training completed "
        f"(epochs: {self.public_training_epochs}, lr: {self.public_training_learning_rate:.4f}, loss: {train_loss:.4f})"
      )

      # Step 3: Generate server logits using trained model
      self.server_generated_logits = self._generate_server_logits(server_round)

      # Save server model state after training for next round
      self._save_server_model_state(server_round)
    else:
      print(f"[FedKD-PublicDist] Round {server_round}: No valid logits received from clients")

    # 通信コストを記録
    self.communication_costs["client_to_server_logits_mb"].append(total_logits_mb)

    # ラウンドの総通信コストを計算
    server_to_client_logits_mb = self.communication_costs["server_to_client_logits_mb"][-1] if self.communication_costs["server_to_client_logits_mb"] else 0.0
    total_round_mb = server_to_client_logits_mb + total_logits_mb
    self.communication_costs["total_round_mb"].append(total_round_mb)

    print(
      f"[FedKD-PublicDist] Round {server_round}: Server->Client: {server_to_client_logits_mb:.4f} MB, "
      f"Client->Server logits: {total_logits_mb:.4f} MB, total: {total_round_mb:.4f} MB"
    )

    # メトリクスの集約
    aggregated_metrics = {}
    if self.fit_metrics_aggregation_fn:
      fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
      aggregated_metrics = self.fit_metrics_aggregation_fn(fit_metrics)

    # 現在の温度をメトリクスに追加
    aggregated_metrics["current_kd_temperature"] = self.kd_temperature

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
    }

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

    # サーバーで生成されたロジットがある場合のみ追加
    if self.server_generated_logits:
      config["avg_logits"] = batch_list_to_base64(self.server_generated_logits)

    # サーバーモデルのパラメータを取得してクライアントに送信
    if self.server_model is not None:
      ndarrays = [val.cpu().numpy() for _, val in self.server_model.state_dict().items()]
      parameters = ndarrays_to_parameters(ndarrays)
      print(f"[FedKD-PublicDist] Round {server_round}: Sending server model parameters to clients for evaluation")
    else:
      print(f"[FedKD-PublicDist] Round {server_round}: No server model available, using provided parameters")

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
      print(f"[FedKD-PublicDist] Round {server_round} - Accuracy: {accuracy:.4f}, Loss: {loss_aggregated:.4f}")

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

    This FedKD implementation with server-side model training performs knowledge distillation
    using client-aggregated logits and generates new logits using the trained server model.

    Parameters
    ----------
    server_round : int
        The current round of federated learning.
    parameters: Parameters
        The current (global) model parameters (unused in FedKD with server model).

    Returns
    -------
    evaluation_result : Optional[Tuple[float, Dict[str, Scalar]]]
        Always returns None as FedKD focuses on logit-based knowledge distillation.
    """
    # FedKDはロジットベースの知識蒸留を使用するため、
    # サーバー側でのパラメータベース評価は行わない
    return None
