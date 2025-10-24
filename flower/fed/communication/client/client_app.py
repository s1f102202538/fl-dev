from fed.data.data_loader_config import DataLoaderConfig
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
  dataset_name = str(context.run_config["dataset_name"])
  local_epochs = context.run_config["local-epochs"]
  partition_id = int(context.node_config["partition-id"])
  num_partitions = int(context.node_config["num-partitions"])

  # MOONパラメータ（オプション）
  out_dim = int(context.run_config.get("out_dim", 256))
  n_classes = int(context.run_config.get("n_classes", 10))
  use_projection_head = bool(context.run_config.get("use_projection_head", True))

  # 統一されたモデルを使用するかどうか
  unified_model = context.run_config.get("unified_model", False)

  data_loader_config = DataLoaderConfig(dataset_name=dataset_name, partition_id=partition_id, num_partitions=num_partitions)
  train_loader, val_loader = load_data(data_loader_config)

  if client_name == "fed-avg-client":
    # 統一されたモデルを使用する場合（MOONベースのprojection headなし）
    if unified_model:
      net = create_model(model_name, is_moon=True, out_dim=out_dim, n_classes=n_classes, use_projection_head=False)
    else:
      net = create_model(model_name, n_classes=n_classes)

    return FedAvgClient(net, context.state, train_loader, val_loader, local_epochs).to_client()
  elif client_name == "fed-kd-client":
    # 統一されたモデルを使用する場合（MOONベースのprojection headなし）
    if unified_model:
      net = create_model(model_name, is_moon=True, out_dim=out_dim, n_classes=n_classes, use_projection_head=False)
    else:
      net = create_model(model_name, n_classes=n_classes)
    public_test_data = load_public_data(data_loader_config)

    return FedKdClient(net, context.state, train_loader, val_loader, public_test_data, local_epochs).to_client()
  elif client_name == "fed-moon-client":
    # MOON用パラメータを渡す
    net = create_model(model_name, is_moon=True, out_dim=out_dim, n_classes=n_classes, use_projection_head=use_projection_head)
    public_test_data = load_public_data(data_loader_config)

    return FedMoonClient(net, context.state, train_loader, val_loader, public_test_data, local_epochs).to_client()
  else:
    raise ValueError(f"Unknown client name: {client_name}.")


# Flower ClientApp
app = ClientApp(
  client_fn=client_fn,
)
