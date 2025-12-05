"""Federated Learning with Logit Aggregation and Parameter Distribution"""

import json
import os
from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union, override

import torch
import wandb
from fed.algorithms.distillation import Distillation
from fed.algorithms.loca_based_distillation import LoCaBasedDistillation
from fed.models.base_model import BaseModel
from fed.util.communication_cost import calculate_communication_cost, calculate_data_size_mb
from fed.util.model_util import base64_to_batch_list, create_run_dir
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
from flwr.server.strategy.aggregate import weighted_loss_avg
from torch import Tensor
from torch.utils.data import DataLoader


class FedKDParamsShare(Strategy):
  """Federated Learning strategy with logit aggregation and parameter distribution.

  This strategy combines:
  1. Server-side: Send model parameters to clients
  2. Client-side: Train locally and generate logits from trained model
  3. Client-side: Send logits back to server
  4. Server-side: Aggregate logits and perform knowledge distillation on server model

  Flow:
  - Round 1+: Server sends parameters → Clients train → Clients return logits → Server aggregates logits and distills
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
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.server_model.to(self.device)

    # Aggregated logits from clients (for server-side distillation)
    self.aggregated_logits: List[Tensor] = []

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
      "server_to_client_params_mb": [],
      "client_to_server_logits_mb": [],
      "total_round_mb": [],
    }

  def _init_wandb_project(self) -> None:
    """Initialize W&B project."""
    wandb_project_name = os.getenv("WANDB_PROJECT_NAME", "federated-learning-default")
    wandb.init(project=wandb_project_name, name=f"{str(self.run_dir)}-ServerApp-FedKDParamsShare")
    print(f"[W&B] Initialized project: {wandb_project_name}")

  def _aggregate_logits(self, client_logits_list: List[List[Tensor]], weights: List[int]) -> List[Tensor]:
    """Aggregate logits from multiple clients using simple average."""
    num_clients = len(client_logits_list)
    num_batches = len(client_logits_list[0])

    aggregated = []
    for batch_idx in range(num_batches):
      batch_sum = None
      for client_logits in client_logits_list:
        if batch_idx < len(client_logits):
          if batch_sum is None:
            batch_sum = client_logits[batch_idx]
          else:
            batch_sum = batch_sum + client_logits[batch_idx]
      if batch_sum is not None:
        # Simple average: divide by number of clients
        aggregated.append(batch_sum / num_clients)

    return aggregated

  def _distill_server_model(self, server_round: int) -> None:
    """Perform knowledge distillation on server model using aggregated client logits."""

    distillation = Distillation(
      studentModel=self.server_model,
      public_data=self.public_data_loader,
      soft_targets=self.aggregated_logits,
    )

    self.server_model = distillation.train_knowledge_distillation(
      epochs=20,
      learning_rate=0.001,
      T=self.kd_temperature,
      alpha=0.7,
      beta=0.3,
      device=self.device,
    )
    print(f"[FedKD-ParamsShare] Round {server_round}: Server model distillation completed")

  def store_results_and_log(self, server_round: int, tag: str, results_dict: Dict) -> None:
    """A helper method that stores results and logs them to W&B if enabled."""
    # Store results
    if tag not in self.results:
      self.results[tag] = {}
    self.results[tag][server_round] = results_dict

    # Save to file
    with open(f"{self.save_path}/results.json", "w", encoding="utf-8") as fp:
      json.dump(self.results, fp, indent=2)

    # Log to W&B (without tag prefix to match other strategies)
    if self.use_wandb:
      wandb.log(results_dict, step=server_round)

  @override
  def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
    """Initialize global model parameters to send to clients."""
    print("[FedKD-ParamsShare] Initializing server model parameters")

    # Return initial model parameters
    initial_parameters = self.initial_parameters

    # Debug: Log initial parameter statistics
    if initial_parameters is not None:
      initial_ndarrays = parameters_to_ndarrays(initial_parameters)
      if len(initial_ndarrays) > 0:
        first_layer_mean = float(initial_ndarrays[0].mean())
        first_layer_std = float(initial_ndarrays[0].std())
        first_layer_sum = float(initial_ndarrays[0].sum())
        print(f"[FedKD-ParamsShare] INITIAL parameters - first layer mean: {first_layer_mean:.6f}, std: {first_layer_std:.6f}, sum: {first_layer_sum:.6f}")

    self.initial_parameters = None
    return initial_parameters

  @override
  def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy, FitIns]]:
    """Configure the next round of training by sending model parameters to clients."""
    config: Dict[str, Scalar] = {"temperature": self.kd_temperature}
    if self.on_fit_config_fn is not None:
      custom_config = self.on_fit_config_fn(server_round)
      config.update(custom_config)
      config["temperature"] = self.kd_temperature

    # Send server model parameters to clients
    ndarrays = [val.cpu().numpy() for _, val in self.server_model.state_dict().items()]
    parameters = ndarrays_to_parameters(ndarrays)

    # Calculate parameter size (server -> client communication)
    self.last_params_size_mb = calculate_communication_cost(parameters)["size_mb"]

    print(f"[FedKD-ParamsShare] Round {server_round}: Sending model parameters to clients (size: {self.last_params_size_mb:.4f} MB)")

    fit_ins = FitIns(parameters, config)

    # Sample clients
    sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
    clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)

    # Return client/config pairs
    return [(client, fit_ins) for client in clients]

  @override
  def aggregate_fit(
    self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
  ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
    """Aggregate client logits and perform knowledge distillation on server model."""
    if not results:
      return None, {}

    # Do not aggregate if there are failures and failures are not accepted
    if not self.accept_failures and failures:
      return None, {}

    # Extract logits from client results
    client_logits_list = []
    client_weights = []

    for _, fit_res in results:
      if "logits" in fit_res.metrics:
        logits_base64 = str(fit_res.metrics["logits"])
        logits = base64_to_batch_list(logits_base64)
        client_logits_list.append(logits)
        client_weights.append(fit_res.num_examples)

    if not client_logits_list:
      print(f"[FedKD-ParamsShare] Round {server_round}: No logits received from clients")
      return None, {}

    # Calculate communication costs
    # Parameter size: from configure_fit (server -> client)
    params_size_mb = getattr(self, "last_params_size_mb", 0.0)

    # Logit size: calculate from received logits (client -> server)
    logits_size_mb = sum(calculate_data_size_mb(str(fit_res.metrics.get("logits", ""))) for _, fit_res in results if "logits" in fit_res.metrics) / len(
      client_logits_list
    )
    total_size_mb = params_size_mb + logits_size_mb

    self.communication_costs["server_to_client_params_mb"].append(params_size_mb)
    self.communication_costs["client_to_server_logits_mb"].append(logits_size_mb)
    self.communication_costs["total_round_mb"].append(total_size_mb)

    print(
      f"[FedKD-ParamsShare] Round {server_round}: Server->Client params: {params_size_mb:.4f} MB, Client->Server logits: {logits_size_mb:.4f} MB, total: {total_size_mb:.4f} MB"
    )

    # Aggregate logits using weighted average
    self.aggregated_logits = self._aggregate_logits(client_logits_list, client_weights)
    print(f"[FedKD-ParamsShare] Round {server_round}: Aggregated {len(self.aggregated_logits)} logit batches from {len(client_logits_list)} clients")

    # Perform knowledge distillation on server model using aggregated logits
    self._distill_server_model(server_round)

    # Get updated server model parameters
    updated_ndarrays = [val.cpu().numpy() for _, val in self.server_model.state_dict().items()]

    # Debug: Log first layer statistics
    first_layer_mean = float(updated_ndarrays[0].mean())
    first_layer_std = float(updated_ndarrays[0].std())
    first_layer_sum = float(updated_ndarrays[0].sum())
    print(
      f"[FedKD-ParamsShare] Round {server_round}: Server model after distillation - first layer mean: {first_layer_mean:.6f}, std: {first_layer_std:.6f}, sum: {first_layer_sum:.6f}"
    )

    # Aggregate custom metrics if aggregation function is provided
    metrics_aggregated = {}
    if self.fit_metrics_aggregation_fn:
      fit_metrics = [(fit_res.num_examples, fit_res.metrics) for _, fit_res in results]
      metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

    # Add communication cost metrics
    metrics_aggregated["comm_cost_server_to_client_params_mb"] = params_size_mb
    metrics_aggregated["comm_cost_client_to_server_logits_mb"] = logits_size_mb
    metrics_aggregated["comm_cost_total_round_mb"] = total_size_mb
    metrics_aggregated["comm_cost_cumulative_mb"] = sum(self.communication_costs["total_round_mb"])

    # Log to W&B if enabled
    self.store_results_and_log(server_round, "fit", metrics_aggregated)

    return ndarrays_to_parameters(updated_ndarrays), metrics_aggregated

  @override
  def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy, EvaluateIns]]:
    """Configure the next round of evaluation."""
    config: Dict[str, Scalar] = {"current_round": server_round}
    if self.on_evaluate_config_fn is not None:
      custom_config = self.on_evaluate_config_fn(server_round)
      config.update(custom_config)

    # Send server model parameters to clients for evaluation
    ndarrays = [val.cpu().numpy() for _, val in self.server_model.state_dict().items()]
    parameters = ndarrays_to_parameters(ndarrays)

    # Debug: Log first layer statistics
    first_layer_mean = float(ndarrays[0].mean())
    first_layer_std = float(ndarrays[0].std())
    first_layer_sum = float(ndarrays[0].sum())
    print(
      f"[FedKD-ParamsShare] Round {server_round}: Sending server model parameters for evaluation - first layer mean: {first_layer_mean:.6f}, std: {first_layer_std:.6f}, sum: {first_layer_sum:.6f}"
    )

    evaluate_ins = EvaluateIns(parameters, config)

    # Sample clients
    sample_size, min_num_clients = self.num_evaluate_clients(client_manager.num_available())
    clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)

    return [(client, evaluate_ins) for client in clients]

  @override
  def aggregate_evaluate(
    self, server_round: int, results: List[Tuple[ClientProxy, EvaluateRes]], failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]
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
    self.store_results_and_log(server_round, "federated_evaluate", {"federated_evaluate_loss": loss_aggregated, **metrics_aggregated})

    return loss_aggregated, metrics_aggregated

  @override
  def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
    """Evaluate the aggregated model parameters (optional centralized evaluation)."""
    # This strategy focuses on federated evaluation
    return None

  def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
    """Return the sample size and the required number of clients."""
    num_clients = int(num_available_clients * self.fraction_fit)
    return max(num_clients, self.min_fit_clients), self.min_available_clients

  def num_evaluate_clients(self, num_available_clients: int) -> Tuple[int, int]:
    """Return the sample size and the required number of clients."""
    num_clients = int(num_available_clients * self.fraction_evaluate)
    return max(num_clients, self.min_evaluate_clients), self.min_available_clients
