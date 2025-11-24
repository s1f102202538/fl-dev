"""pytorch-example: A Flower / PyTorch app."""

import json
import os
from logging import INFO
from typing import Dict, List, Optional, Union

import wandb
from fed.models.base_model import BaseModel
from fed.util.communication_cost import calculate_communication_cost
from fed.util.model_util import create_run_dir
from flwr.common import EvaluateRes, Scalar, logger
from flwr.common.typing import Parameters, UserConfig
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


class CustomFedAvg(FedAvg):
  def __init__(self, net: BaseModel, run_config: UserConfig, use_wandb: bool, *args: object, **kwargs: object) -> None:
    super().__init__(*args, **kwargs)  # type: ignore

    # Create a directory where to save results from this run
    self.net = net
    self.save_path, self.run_dir = create_run_dir(run_config)
    self.use_wandb = use_wandb

    # Store model creation parameters for proper model initialization
    self.model_config = run_config

    # Initialise W&B if set
    if use_wandb:
      self._init_wandb_project()

    # A dictionary to store results as they come
    self.results: Dict = {}

    # 通信コスト追跡用の変数
    self.communication_costs: Dict[str, List[float]] = {
      "server_to_client_params_mb": [],  # サーバからクライアントへのパラメータ送信コスト
      "client_to_server_params_mb": [],  # クライアントからサーバへのパラメータ送信コスト
      "total_round_mb": [],  # ラウンドごとの総通信コスト
    }

  def _init_wandb_project(self) -> None:
    """Initialize W&B project."""
    wandb_project_name = os.getenv("WANDB_PROJECT_NAME", "federated-learning-default")
    wandb.init(project=wandb_project_name, name=f"{str(self.run_dir)}-ServerApp-FedAvg")

  def store_results_and_log(self, server_round: int, tag: str, results_dict: Dict) -> None:
    """A helper method that stores results and logs them to W&B if enabled."""

    if self.use_wandb:
      # Log metrics to W&B
      wandb.log(results_dict, step=server_round)

  def configure_fit(self, server_round: int, parameters: Parameters, client_manager) -> List:
    """Configure the next round of training with communication cost measurement."""
    # 基底クラスのconfigure_fitを呼び出してクライアント設定を取得
    config_list = super().configure_fit(server_round, parameters, client_manager)

    # 送信パラメータのサイズを測定
    comm_cost = calculate_communication_cost(parameters)
    # 実際に送信される総量は (パラメータサイズ × クライアント数)
    num_clients = len(config_list)
    total_server_to_client_mb = comm_cost["size_mb"] * num_clients
    self.communication_costs["server_to_client_params_mb"].append(total_server_to_client_mb)

    logger.log(
      INFO,
      f"Round {server_round}: Server->Client parameters: {comm_cost['size_mb']:.4f} MB per client, total: {total_server_to_client_mb:.4f} MB ({num_clients} clients)",
    )

    return config_list

  def aggregate_fit(self, server_round: int, results, failures):
    """Aggregate training results with communication cost measurement."""
    # クライアントからのパラメータサイズを測定
    total_params_mb = 0.0

    for _, fit_res in results:
      # パラメータサイズ測定
      if fit_res.parameters:
        params_cost = calculate_communication_cost(fit_res.parameters)
        total_params_mb += params_cost["size_mb"]

    # 通信コストを記録
    self.communication_costs["client_to_server_params_mb"].append(total_params_mb)

    # ラウンドの総通信コストを計算（メトリクスは除外）
    server_to_client_params_mb = self.communication_costs["server_to_client_params_mb"][-1] if self.communication_costs["server_to_client_params_mb"] else 0.0
    total_round_mb = server_to_client_params_mb + total_params_mb
    self.communication_costs["total_round_mb"].append(total_round_mb)

    logger.log(INFO, f"Round {server_round}: Client->Server params: {total_params_mb:.4f} MB, total: {total_round_mb:.4f} MB")

    # 基底クラスのaggregate_fitを呼び出し
    parameters, metrics = super().aggregate_fit(server_round, results, failures)

    # 通信コストをメトリクスに追加
    if metrics is not None:
      metrics["comm_cost_server_to_client_mb"] = server_to_client_params_mb
      metrics["comm_cost_client_to_server_params_mb"] = total_params_mb
      metrics["comm_cost_total_round_mb"] = total_round_mb
      metrics["comm_cost_cumulative_mb"] = sum(self.communication_costs["total_round_mb"])

    # 通信コストメトリクスをW&Bにログ
    communication_metrics = {
      "comm_cost_server_to_client_mb": server_to_client_params_mb,
      "comm_cost_client_to_server_params_mb": total_params_mb,
      "comm_cost_total_round_mb": total_round_mb,
      "comm_cost_cumulative_mb": sum(self.communication_costs["total_round_mb"]),
    }

    self.store_results_and_log(server_round=server_round, tag="communication_costs", results_dict=communication_metrics)

    return parameters, metrics

  def evaluate(self, server_round: int, parameters: Parameters) -> Optional[tuple[float, dict[str, Scalar]]]:
    """Run centralized evaluation if callback was passed to strategy init."""
    loss, metrics = super().evaluate(server_round, parameters)  # type: ignore

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
