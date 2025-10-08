from typing import Callable, Tuple

import torch
from datasets import load_dataset
from flwr.common import Context, ndarrays_to_parameters
from flwr.common.typing import NDArrays, UserConfig
from flwr.server import ServerAppComponents, ServerConfig
from torch.utils.data import DataLoader

from fed.data.data_loader_manager import DataLoaderManager
from fed.data.data_transform_manager import DataTransformManager
from fed.models.mini_cnn import MiniCNN
from fed.task.cnn_task import CNNTask
from fed.util.model_util import get_weights, set_weights, weighted_average

from ..strategy.fed_avg import CustomFedAvg


class FedAvgServer:
  @staticmethod
  def gen_evaluate_fn(
    testloader: DataLoader,
    device: torch.device,
  ) -> Callable:
    """Generate the function for centralized evaluation."""

    def evaluate(server_round: int, parameters_ndarrays: NDArrays, config: UserConfig) -> Tuple[float, object]:
      """Evaluate global model on centralized test set."""
      net = MiniCNN()
      set_weights(net, parameters_ndarrays)
      net.to(device)
      loss, accuracy = CNNTask.test(net, testloader, device=device)
      return loss, {"centralized_accuracy": accuracy}

    return evaluate

  @staticmethod
  def on_fit_config(server_round: int) -> object:
    """Construct `config` that clients receive when running `fit()`"""
    lr = 0.1
    # Enable a simple form of learning rate decay
    if server_round > 10:
      lr /= 2
    return {"lr": lr}

  @staticmethod
  def server_fn(context: Context) -> ServerAppComponents:
    # Read from config
    num_rounds = int(context.run_config["num-server-rounds"])
    fraction_fit = context.run_config["fraction-fit"]
    fraction_eval = context.run_config["fraction-evaluate"]
    server_device = torch.device(str(context.run_config["server-device"]))

    # Initialize model parameters
    ndarrays = get_weights(MiniCNN())
    parameters = ndarrays_to_parameters(ndarrays)

    # Prepare dataset for central evaluation

    # This is the exact same dataset as the one downloaded by the clients via
    # FlowerDatasets. However, we don't use FlowerDatasets for the server since
    # partitioning is not needed.
    # We make use of the "test" split only
    global_test_set = load_dataset("zalando-datasets/fashion_mnist", split="test")

    testloader = DataLoader(
      global_test_set.with_transform(DataTransformManager(DataLoaderManager()).apply_eval_transforms),  # type: ignore
      batch_size=32,
    )

    # Define strategy
    strategy = CustomFedAvg(
      run_config=context.run_config,
      use_wandb=bool(context.run_config["use-wandb"]),
      fraction_fit=fraction_fit,
      fraction_evaluate=fraction_eval,
      initial_parameters=parameters,
      on_fit_config_fn=FedAvgServer.on_fit_config,
      evaluate_fn=FedAvgServer.gen_evaluate_fn(testloader, device=server_device),
      evaluate_metrics_aggregation_fn=weighted_average,
      min_fit_clients=5,
      min_evaluate_clients=5,
      min_available_clients=5,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)
