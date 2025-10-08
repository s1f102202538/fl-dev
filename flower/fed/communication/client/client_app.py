from fed.util.create_model import create_model
from fed.util.data_loader import load_data, load_public_data
from flwr.client import ClientApp
from flwr.client.client import Client
from flwr.common import Context

from .apps.fed_avg_client import FedAvgClient
from .apps.fed_kd_client import FedKdClient
from .apps.fed_moon_client import FedMoonClient


def client_fn(context: Context) -> Client:
  model_name = str(context.run_config["model_name"])
  client_name = str(context.run_config["client_name"])
  local_epochs = context.run_config["local-epochs"]
  partition_id = context.node_config["partition-id"]
  num_partitions = context.node_config["num-partitions"]
  train_loader, val_loader = load_data(partition_id, num_partitions)

  if client_name == "fed-avg-client":
    net = create_model(model_name)

    return FedAvgClient(net, context.state, train_loader, val_loader, local_epochs).to_client()
  elif client_name == "fed-kd-client":
    net = create_model(model_name)
    public_test_data = load_public_data(batch_size=32, max_samples=1000)

    return FedKdClient(net, context.state, train_loader, val_loader, public_test_data, local_epochs).to_client()
  elif client_name == "fed-moon-client":
    net = create_model(model_name, is_moon=True)
    public_test_data = load_public_data(batch_size=32, max_samples=1000)

    return FedMoonClient(net, context.state, train_loader, val_loader, public_test_data, local_epochs).to_client()
  else:
    raise ValueError(f"Unknown client name: {client_name}.")


# Flower ClientApp
app = ClientApp(
  client_fn=client_fn,
)
