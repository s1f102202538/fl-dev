"""Federated Learning with Parameter Aggregation and Knowledge Distillation via Logits"""

import json
import os
from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union, override

import torch
import wandb
from fed.models.base_model import BaseModel
from fed.task.cnn_task import CNNTask
from fed.util.communication_cost import calculate_communication_cost, calculate_data_size_mb
from fed.util.model_util import batch_list_to_base64, create_run_dir, set_weights
from flwr.common import (
  EvaluateIns,
  EvaluateRes,
  FitIns,
  FitRes,
  MetricsAggregationFn,
  Parameters,
  Scalar,
  ndarrays_to_parameters,
  parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.common.typing import UserConfig
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from torch import Tensor
from torch.utils.data import DataLoader


class FedKDParamsShare(Strategy):
  """Federated Learning strategy with parameter aggregation and logit-based knowledge distillation.

  This strategy combines:
  1. Client-side: Knowledge distillation using server-provided logits (from Round 2+)
  2. Client-side: Local training with MOON contrastive learning
  3. Server-side: Parameter aggregation (FedAvg-style weighted averaging)
  4. Server-side: Logit generation from aggregated model for next round

  Flow:
  - Round 1: No logits sent → Clients train locally → Return parameters
  - Round 2+: Server aggregates parameters → Generates logits → Clients distill and train
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
    kd_temperature: float = 3.0,
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

    # Server-side model and data
    self.server_model = server_model
    self.public_data_loader = public_data_loader
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.server_model.to(self.device)

    # Server-generated logits for distribution to clients
    self.server_generated_logits: List[Tensor] = []

    # Knowledge distillation temperature
    self.kd_temperature = kd_temperature

    self.save_path, self.run_dir = create_run_dir(run_config)
    self.use_wandb = use_wandb

    # Initialize W&B if enabled
    if use_wandb:
      self._init_wandb_project()

    # Store results
    self.results: Dict = {}

    # Communication cost tracking
    self.communication_costs: Dict[str, List[float]] = {
      "server_to_client_logits_mb": [],
      "client_to_server_params_mb": [],
      "total_round_mb": [],
    }

  def _init_wandb_project(self) -> None:
    """Initialize Weights & Biases project for experiment tracking."""
    wandb_project_name = os.environ.get("WANDB_PROJECT_NAME", "federated-learning")
    wandb.init(project=wandb_project_name, name=f"FedKDParamsShare-{self.run_dir}")
    print(f"[W&B] Initialized project: {wandb_project_name}")

  def _generate_logits_from_aggregated_model(self) -> List[Tensor]:
    """Generate logits from the aggregated server model."""
    logits = CNNTask.inference(self.server_model, self.public_data_loader, device=self.device)
    print(f"[FedKD-ParamsShare] Generated {len(logits)} logit batches from aggregated model")
    return logits

  def store_results_and_log(self, server_round: int, tag: str, results_dict: Dict) -> None:
    """Store results to filesystem and log to W&B if enabled."""
    if tag not in self.results:
      self.results[tag] = {}
    self.results[tag][server_round] = results_dict

    # Save to file
    with open(f"{self.save_path}/results.json", "w", encoding="utf-8") as fp:
      json.dump(self.results, fp, indent=2)

    # Log to W&B
    if self.use_wandb:
      wandb.log({f"{tag}/{key}": value for key, value in results_dict.items()}, step=server_round)

  @override
  def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
    """Initialize global model parameters without generating logits for Round 1."""
    # Round 1: No logits distribution - clients will train without distillation
    self.server_generated_logits = []
    print("[FedKD-ParamsShare] Round 1: No logits will be sent to clients")

    # Return initial model parameters
    initial_parameters = self.initial_parameters
    self.initial_parameters = None
    return initial_parameters

  @override
  def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy, FitIns]]:
    """Configure the next round of training."""
    config = {}
    if self.on_fit_config_fn is not None:
      config = self.on_fit_config_fn(server_round)

    # Add temperature parameter
    config["temperature"] = self.kd_temperature

    # サーバーからクライアントへの通信コスト測定
    server_to_client_mb = 0.0

    # Add server-generated logits for knowledge distillation
    if self.server_generated_logits:
      logits_data = batch_list_to_base64(self.server_generated_logits)
      config["avg_logits"] = logits_data
      # ロジットデータのサイズを測定
      server_to_client_mb = calculate_data_size_mb(logits_data)
      print(
        f"[FedKD-ParamsShare] Round {server_round}: Sending {len(self.server_generated_logits)} logit batches (temp: {self.kd_temperature:.3f}, size: {server_to_client_mb:.4f} MB)"
      )
    else:
      print(f"[FedKD-ParamsShare] Round {server_round}: No logits available for this round")

    fit_ins = FitIns(parameters, config)

    # Sample clients
    sample_size = int(self.fraction_fit * client_manager.num_available())
    clients = client_manager.sample(num_clients=sample_size, min_num_clients=self.min_fit_clients)

    # 実際の通信コストはクライアント数を考慮
    total_server_to_client_mb = server_to_client_mb * len(clients)
    self.communication_costs["server_to_client_logits_mb"].append(total_server_to_client_mb)

    print(f"[FedKD-ParamsShare] Round {server_round}: Total server->client communication: {total_server_to_client_mb:.4f} MB ({len(clients)} clients)")

    return [(client, fit_ins) for client in clients]

  @override
  def aggregate_fit(
    self,
    server_round: int,
    results: List[Tuple[ClientProxy, FitRes]],
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
  ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
    """Aggregate model parameters using FedAvg-style weighted averaging."""
    if not results:
      return None, {}
    if not self.accept_failures and failures:
      return None, {}

    # Calculate client->server communication cost
    total_params_mb = 0.0
    for _, fit_res in results:
      if fit_res.parameters.tensors:
        params_mb = calculate_communication_cost(fit_res.parameters)
        total_params_mb += params_mb["size_mb"]

    self.communication_costs["client_to_server_params_mb"].append(total_params_mb)

    # Calculate total round communication cost
    server_to_client_mb = self.communication_costs["server_to_client_logits_mb"][-1] if self.communication_costs["server_to_client_logits_mb"] else 0.0
    total_round_mb = server_to_client_mb + total_params_mb
    self.communication_costs["total_round_mb"].append(total_round_mb)

    print(
      f"[FedKD-ParamsShare] Round {server_round}: Server->Client: {server_to_client_mb:.4f} MB, Client->Server params: {total_params_mb:.4f} MB, total: {total_round_mb:.4f} MB"
    )

    # Aggregate parameters using weighted average
    weights_results = [(parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples) for _, fit_res in results]
    aggregated_ndarrays = aggregate(weights_results)
    parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

    # Update server model with aggregated parameters
    set_weights(self.server_model, aggregated_ndarrays)
    print(f"[FedKD-ParamsShare] Round {server_round}: Server model updated with aggregated parameters")

    # Generate new logits from aggregated model for next round
    self.server_generated_logits = self._generate_logits_from_aggregated_model()

    # Aggregate custom metrics
    metrics_aggregated = {}
    if self.fit_metrics_aggregation_fn:
      fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
      metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

    # Add communication cost metrics
    metrics_aggregated["comm_cost_server_to_client_mb"] = server_to_client_mb
    metrics_aggregated["comm_cost_client_to_server_params_mb"] = total_params_mb
    metrics_aggregated["comm_cost_total_round_mb"] = total_round_mb
    metrics_aggregated["comm_cost_cumulative_mb"] = sum(self.communication_costs["total_round_mb"])

    # Log communication costs
    communication_metrics = {
      "comm_cost_server_to_client_mb": server_to_client_mb,
      "comm_cost_client_to_server_params_mb": total_params_mb,
      "comm_cost_total_round_mb": total_round_mb,
      "comm_cost_cumulative_mb": sum(self.communication_costs["total_round_mb"]),
    }

    self.store_results_and_log(server_round=server_round, tag="communication_costs", results_dict=communication_metrics)

    return parameters_aggregated, metrics_aggregated

  @override
  def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy, EvaluateIns]]:
    """Configure the next round of evaluation."""
    config = {}
    config["current_round"] = server_round

    # Send aggregated model parameters for evaluation
    if self.server_model is not None:
      ndarrays = [val.cpu().numpy() for _, val in self.server_model.state_dict().items()]
      parameters = ndarrays_to_parameters(ndarrays)
      print(f"[FedKD-ParamsShare] Round {server_round}: Sending aggregated model parameters to clients for evaluation")

    evaluate_ins = EvaluateIns(parameters, config)

    # Sample clients
    sample_size = int(self.fraction_evaluate * client_manager.num_available())
    clients = client_manager.sample(num_clients=sample_size, min_num_clients=self.min_evaluate_clients)

    return [(client, evaluate_ins) for client in clients]

  @override
  def aggregate_evaluate(
    self,
    server_round: int,
    results: List[Tuple[ClientProxy, EvaluateRes]],
    failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
  ) -> Tuple[Optional[float], Dict[str, Scalar]]:
    """Aggregate evaluation results."""
    if not results:
      return None, {}
    if not self.accept_failures and failures:
      return None, {}

    # Aggregate loss
    loss_aggregated = weighted_loss_avg([(evaluate_res.num_examples, evaluate_res.loss) for _, evaluate_res in results])

    # Aggregate custom metrics
    metrics_aggregated = {}
    if self.evaluate_metrics_aggregation_fn:
      eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
      metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
    elif server_round == 1:
      log(WARNING, "No evaluate_metrics_aggregation_fn provided")

    # Log accuracy information
    if "accuracy" in metrics_aggregated:
      accuracy = metrics_aggregated["accuracy"]
      print(f"[FedKD-ParamsShare] Round {server_round}: Federated evaluation accuracy: {accuracy:.4f}")

    # Store and log results
    self.store_results_and_log(
      server_round=server_round,
      tag="federated_evaluate",
      results_dict={"federated_evaluate_loss": loss_aggregated, **metrics_aggregated},
    )

    return loss_aggregated, metrics_aggregated

  @override
  def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
    """Evaluate the aggregated model parameters (optional centralized evaluation)."""
    # This strategy focuses on federated evaluation
    # Centralized evaluation can be added if needed
    return None
