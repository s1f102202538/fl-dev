from flwr.common import Context
from flwr.server import ServerApp, ServerAppComponents
from torch import device

from .apps.fed_avg_server import FedAvgServer
from .apps.fed_kd_server import FedKDServer


def server_fn(context: Context) -> ServerAppComponents:
  model_name = str(context.run_config["model_name"])
  dataset_name = str(context.run_config["dataset_name"])
  server_name = str(context.run_config["server_name"])
  use_wandb = bool(context.run_config["use-wandb"])

  num_rounds = int(context.run_config["num-server-rounds"])
  server_device = device(str(context.run_config["server-device"]))

  if server_name == "fed-avg-server":
    return FedAvgServer.create_server(model_name, dataset_name, use_wandb, context.run_config, server_device, num_rounds)
  elif server_name == "fed-kd-server":
    return FedKDServer.create_server(use_wandb, context.run_config, num_rounds)
  else:
    raise ValueError(f"Unknown server name: {server_name}.")


# Create ServerApp
app = ServerApp(server_fn=server_fn)
