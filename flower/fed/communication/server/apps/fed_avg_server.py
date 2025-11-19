from typing import Callable, Tuple

from fed.models.base_model import BaseModel
from fed.task.cnn_task import CNNTask
from fed.util.model_util import get_weights, set_weights, weighted_average
from flwr.common import ndarrays_to_parameters
from flwr.common.typing import NDArrays, UserConfig
from flwr.server import ServerAppComponents, ServerConfig
from torch import device
from torch.utils.data import DataLoader

from ..strategy.fed_avg import CustomFedAvg


class FedAvgServer:
  @staticmethod
  def gen_evaluate_fn(testloader: DataLoader, device: device, net: BaseModel) -> Callable:
    """Generate the function for centralized evaluation."""

    def evaluate(server_round: int, parameters_ndarrays: NDArrays, config: UserConfig) -> Tuple[float, object]:
      """Evaluate global model on centralized test set."""
      set_weights(net, parameters_ndarrays)
      net.to(device)
      loss, accuracy = CNNTask.test(net, testloader, device=device)
      return loss, {"centralized_accuracy": accuracy}

    return evaluate

  @staticmethod
  def on_fit_config(server_round: int) -> object:
    """Construct `config` that clients receive when running `fit()`"""
    lr = 0.01

    return {"lr": lr}

  @staticmethod
  def create_server(net: BaseModel, use_wandb: bool, run_config, server_device: device, num_rounds: int, testloader: DataLoader) -> ServerAppComponents:
    parameters = ndarrays_to_parameters(get_weights(net))

    # Define strategy
    strategy = CustomFedAvg(
      net=net,
      run_config=run_config,
      use_wandb=use_wandb,
      fraction_fit=1.0,
      fraction_evaluate=1.0,
      initial_parameters=parameters,
      on_fit_config_fn=FedAvgServer.on_fit_config,
      evaluate_fn=FedAvgServer.gen_evaluate_fn(testloader, server_device, net),
      evaluate_metrics_aggregation_fn=weighted_average,
      min_fit_clients=5,
      min_evaluate_clients=5,
      min_available_clients=5,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)
