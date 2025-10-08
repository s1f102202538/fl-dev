"""pytorch-example: A Flower / PyTorch app."""

import json
import os
from logging import INFO
from typing import Dict, List, Optional, Union

import torch
import wandb
from fed.models.base_model import BaseModel
from fed.util.communication_cost import calculate_communication_cost, calculate_metrics_communication_cost
from fed.util.model_util import create_run_dir, set_weights
from flwr.common import EvaluateRes, Scalar, logger, parameters_to_ndarrays
from flwr.common.typing import Parameters, UserConfig
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

WANDB_PROJECT_NAME = os.getenv("WANDB_PROJECT_NAME")


class CustomFedAvg(FedAvg):
  """A class that behaves like FedAvg but has extra functionality.

  This strategy: (1) saves results to the filesystem, (2) saves a
  checkpoint of the global  model when a new best is found, (3) logs
  results to W&B if enabled.
  """

  def __init__(self, net: BaseModel, run_config: UserConfig, use_wandb: bool, *args: object, **kwargs: object) -> None:
    super().__init__(*args, **kwargs)  # type: ignore

    # Create a directory where to save results from this run
    self.net = net
    self.save_path, self.run_dir = create_run_dir(run_config)
    self.use_wandb = use_wandb
    # Initialise W&B if set
    if use_wandb:
      self._init_wandb_project()

    # Keep track of best acc
    self.best_acc_so_far = 0.0

    # A dictionary to store results as they come
    self.results: Dict = {}

    # 通信コスト追跡用の変数
    self.communication_costs: Dict[str, List[float]] = {
      "server_to_client_params_mb": [],  # サーバからクライアントへのパラメータ送信コスト
      "client_to_server_params_mb": [],  # クライアントからサーバへのパラメータ送信コスト
      "client_to_server_metrics_mb": [],  # クライアントからサーバへのメトリクス送信コスト
      "total_round_mb": [],  # ラウンドごとの総通信コスト
    }

  def _init_wandb_project(self) -> None:
    """Initialize W&B project."""
    wandb.init(project=WANDB_PROJECT_NAME, name=f"{str(self.run_dir)}-ServerApp-FedAvg")

  def _store_results(self, tag: str, results_dict: Dict) -> None:
    """Store results in dictionary, then save as JSON."""
    # Update results dict
    if tag in self.results:
      self.results[tag].append(results_dict)
    else:
      self.results[tag] = [results_dict]

    # Save results to disk.
    # Note we overwrite the same file with each call to this function.
    # While this works, a more sophisticated approach is preferred
    # in situations where the contents to be saved are larger.
    with open(f"{self.save_path}/results.json", "w", encoding="utf-8") as fp:
      json.dump(self.results, fp)

  def _update_best_acc(self, round: int, accuracy: float, parameters: Parameters) -> None:
    """Determines if a new best global model has been found.

    If so, the model checkpoint is saved to disk.
    """
    if accuracy > self.best_acc_so_far:
      self.best_acc_so_far = accuracy
      logger.log(INFO, "💡 New best global model found: %f", accuracy)
      # You could save the parameters object directly.
      # Instead we are going to apply them to a PyTorch
      # model and save the state dict.
      # Converts flwr.common.Parameters to ndarrays
      ndarrays = parameters_to_ndarrays(parameters)
      model = self.net
      set_weights(model, ndarrays)
      # Save the PyTorch model
      file_name = f"model_state_acc_{accuracy}_round_{round}.pth"
      torch.save(model.state_dict(), self.save_path / file_name)

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

  def configure_fit(self, server_round: int, parameters: Parameters, client_manager) -> List:
    """Configure the next round of training with communication cost measurement."""
    # 送信パラメータのサイズを測定
    comm_cost = calculate_communication_cost(parameters)
    self.communication_costs["server_to_client_params_mb"].append(comm_cost["size_mb"])

    logger.log(INFO, f"Round {server_round}: Server->Client parameters: {comm_cost['size_mb']:.4f} MB")

    # 基底クラスのconfigure_fitを呼び出し
    return super().configure_fit(server_round, parameters, client_manager)

  def aggregate_fit(self, server_round: int, results, failures):
    """Aggregate training results with communication cost measurement."""
    # クライアントからのパラメータとメトリクスのサイズを測定
    total_params_mb = 0.0
    total_metrics_mb = 0.0

    for _, fit_res in results:
      # パラメータサイズ測定
      if fit_res.parameters:
        params_cost = calculate_communication_cost(fit_res.parameters)
        total_params_mb += params_cost["size_mb"]

      # メトリクスサイズ測定
      if fit_res.metrics:
        metrics_cost = calculate_metrics_communication_cost(fit_res.metrics)
        total_metrics_mb += metrics_cost["metrics_size_mb"]

    # 通信コストを記録
    self.communication_costs["client_to_server_params_mb"].append(total_params_mb)
    self.communication_costs["client_to_server_metrics_mb"].append(total_metrics_mb)

    # ラウンドの総通信コストを計算
    server_to_client_params_mb = self.communication_costs["server_to_client_params_mb"][-1] if self.communication_costs["server_to_client_params_mb"] else 0.0
    total_round_mb = server_to_client_params_mb + total_params_mb + total_metrics_mb
    self.communication_costs["total_round_mb"].append(total_round_mb)

    logger.log(
      INFO, f"Round {server_round}: Client->Server params: {total_params_mb:.4f} MB, metrics: {total_metrics_mb:.4f} MB, total: {total_round_mb:.4f} MB"
    )

    # 基底クラスのaggregate_fitを呼び出し
    parameters, metrics = super().aggregate_fit(server_round, results, failures)

    # 通信コストをメトリクスに追加
    if metrics is not None:
      metrics["comm_cost_server_to_client_mb"] = server_to_client_params_mb
      metrics["comm_cost_client_to_server_params_mb"] = total_params_mb
      metrics["comm_cost_client_to_server_metrics_mb"] = total_metrics_mb
      metrics["comm_cost_total_round_mb"] = total_round_mb
      metrics["comm_cost_cumulative_mb"] = sum(self.communication_costs["total_round_mb"])

    return parameters, metrics

  def evaluate(self, server_round: int, parameters: Parameters) -> Optional[tuple[float, dict[str, Scalar]]]:
    """Run centralized evaluation if callback was passed to strategy init."""
    loss, metrics = super().evaluate(server_round, parameters)  # type: ignore

    # Save model if new best central accuracy is found
    self._update_best_acc(server_round, float(metrics["centralized_accuracy"]), parameters)

    # Store and log centralized evaluation results
    self.store_results_and_log(
      server_round=server_round,
      tag="fedavg_centralized_evaluate",
      results_dict={"fedavg_centralized_loss": loss, **metrics},
    )
    return loss, metrics

  def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager):
    """Configure the next round of evaluation."""
    # 基底クラスのconfigure_evaluateを呼び出し
    return super().configure_evaluate(server_round, parameters, client_manager)

  def aggregate_evaluate(
    self, server_round: int, results: List[tuple[ClientProxy, EvaluateRes]], failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]]
  ) -> tuple[Optional[float], dict[str, Scalar]]:
    """Aggregate results from federated evaluation."""
    # 基底クラスのaggregate_evaluateを呼び出し
    loss, metrics = super().aggregate_evaluate(server_round, results, failures)

    # Store and log federated evaluation results
    self.store_results_and_log(
      server_round=server_round,
      tag="federated_evaluate",
      results_dict={"federated_evaluate_loss": loss, **metrics},
    )
    return loss, metrics
