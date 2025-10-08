"""pytorch-example: A Flower / PyTorch app."""

import json
from logging import INFO
from typing import Dict, List, Optional, Union

import torch
import wandb
from flwr.common import EvaluateRes, Scalar, logger, parameters_to_ndarrays
from flwr.common.typing import Parameters, UserConfig
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from fl.flower.common.util.model_util import create_run_dir, set_weights
from flower.common.models.mini_cnn import MiniCNN

PROJECT_NAME = "fl-dev-cifer-10"


class CustomFedAvg(FedAvg):
  """A class that behaves like FedAvg but has extra functionality.

  This strategy: (1) saves results to the filesystem, (2) saves a
  checkpoint of the global  model when a new best is found, (3) logs
  results to W&B if enabled.
  """

  def __init__(self, run_config: UserConfig, use_wandb: bool, *args: object, **kwargs: object) -> None:
    super().__init__(*args, **kwargs)  # type: ignore

    # Create a directory where to save results from this run
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
    wandb.init(project=PROJECT_NAME, name=f"{str(self.run_dir)}-ServerApp-FedAvg")

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
      model = MiniCNN()
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

  def aggregate_evaluate(
    self, server_round: int, results: List[tuple[ClientProxy, EvaluateRes]], failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]]
  ) -> tuple[Optional[float], dict[str, Scalar]]:
    """Aggregate results from federated evaluation."""
    loss, metrics = super().aggregate_evaluate(server_round, results, failures)

    # Store and log federated evaluation results
    self.store_results_and_log(
      server_round=server_round,
      tag="federated_evaluate",
      results_dict={"federated_evaluate_loss": loss, **metrics},
    )
    return loss, metrics
