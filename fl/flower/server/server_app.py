"""pytorch-example: A Flower / PyTorch app."""

from typing import Any, Callable, Dict, List, Tuple

import torch
from datasets import load_dataset
from flower.common.models.mini_cnn import MiniCNN
from flower.common.task.cnn_task import CNNTask
from flower.common.util.data_loader import apply_eval_transforms
from flower.common.util.util import get_weights, set_weights
from flower.server.strategy.fed_avg import CustomFedAvg
from flwr.common import Context, ndarrays_to_parameters
from flwr.common.typing import NDArrays, UserConfig
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from torch.utils.data import DataLoader


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


def on_fit_config(server_round: int) -> object:
  """Construct `config` that clients receive when running `fit()`"""
  lr = 0.1
  # Enable a simple form of learning rate decay
  if server_round > 10:
    lr /= 2
  return {"lr": lr}


def weighted_average(metrics: List[Tuple[int, Dict[str, Any]]]) -> object:
  # Multiply accuracy of each client by number of examples used
  accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
  examples = [num_examples for num_examples, _ in metrics]

  # Aggregate and return custom metric (weighted average)
  return {"federated_evaluate_accuracy": sum(accuracies) / sum(examples)}


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
    global_test_set.with_transform(apply_eval_transforms),  # type: ignore
    batch_size=32,
  )

  # Define strategy
  strategy = CustomFedAvg(
    run_config=context.run_config,
    use_wandb=bool(context.run_config["use-wandb"]),
    fraction_fit=fraction_fit,
    fraction_evaluate=fraction_eval,
    initial_parameters=parameters,
    on_fit_config_fn=on_fit_config,
    evaluate_fn=gen_evaluate_fn(testloader, device=server_device),
    evaluate_metrics_aggregation_fn=weighted_average,
  )
  config = ServerConfig(num_rounds=num_rounds)

  return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
