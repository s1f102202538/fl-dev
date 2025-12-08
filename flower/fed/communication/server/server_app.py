import os

from fed.data.data_loader_config import DataLoaderConfig
from fed.util.create_model import create_model
from fed.util.data_loader import load_public_data, load_test_data
from flwr.common import Context
from flwr.server import ServerApp, ServerAppComponents
from torch import device

from .apps.fed_avg_server import FedAvgServer
from .apps.fed_md_avg_server import FedMdAvgServer
from .apps.fed_md_distillation_model_server import FedMdDistillationModelServer
from .apps.fed_md_params_share_csd_server import FedMdParamsShareCsdServer
from .apps.fed_md_params_share_server import FedMdParamsShareServer
from .apps.fed_md_public_distillation_server import FedMdPublicDistillationServer
from .apps.fed_md_weighted_avg_server import FedMdWeightedAvgServer


def server_fn(context: Context) -> ServerAppComponents:
  model_name = str(context.run_config["model_name"])
  dataset_name = str(context.run_config["dataset_name"])
  server_name = str(context.run_config["server_name"])
  wandb_project_name = str(context.run_config["wandb_project_name"])
  n_classes = int(context.run_config.get("n_classes", 10))
  out_dim = int(context.run_config.get("out_dim", 256))
  use_wandb = bool(context.run_config["use-wandb"])

  os.environ["WANDB_PROJECT_NAME"] = wandb_project_name

  num_rounds = int(context.run_config["num-server-rounds"])
  server_device = device(str(context.run_config["server-device"]))

  data_loader_config = DataLoaderConfig(dataset_name=dataset_name)

  testloader = load_test_data(data_loader_config)

  if server_name == "fed-avg-server":
    net = create_model(model_name, is_moon=True, out_dim=out_dim, n_classes=n_classes, use_projection_head=True)
    return FedAvgServer.create_server(net, use_wandb, context.run_config, server_device, num_rounds, testloader)
  elif server_name == "fed-kd-avg-server":
    return FedMdAvgServer.create_server(use_wandb, context.run_config, num_rounds)
  elif server_name == "fed-kd-weighted-avg-server":
    return FedMdWeightedAvgServer.create_server(use_wandb, context.run_config, num_rounds)
  elif server_name == "fed-kd-distillation-model-server":
    # Create server-side model and public data loader here
    server_model = create_model(model_name, is_moon=True, out_dim=out_dim, n_classes=n_classes, use_projection_head=True)
    public_data_loader = load_public_data(data_loader_config)

    return FedMdDistillationModelServer.create_server(
      server_model=server_model,
      public_data_loader=public_data_loader,
      server_device=server_device,
      use_wandb=use_wandb,
      run_config=context.run_config,
      num_rounds=num_rounds,
    )
  elif server_name == "fed-kd-public-distillation-server":
    # Create server-side model and public data loader for public data pre-training
    server_model = create_model(model_name, is_moon=True, out_dim=out_dim, n_classes=n_classes, use_projection_head=True)
    public_data_loader = load_public_data(data_loader_config)

    return FedMdPublicDistillationServer.create_server(
      server_model=server_model,
      public_data_loader=public_data_loader,
      server_device=server_device,
      use_wandb=use_wandb,
      run_config=context.run_config,
      num_rounds=num_rounds,
    )
  elif server_name == "fed-kd-params-share-server":
    # Create server-side model and public data loader for parameter aggregation and logit generation
    server_model = create_model(model_name, is_moon=True, out_dim=out_dim, n_classes=n_classes, use_projection_head=True)
    public_data_loader = load_public_data(data_loader_config)

    return FedMdParamsShareServer.create_server(
      server_model=server_model,
      public_data_loader=public_data_loader,
      server_device=server_device,
      use_wandb=use_wandb,
      run_config=context.run_config,
      num_rounds=num_rounds,
    )
  elif server_name == "fed-kd-params-share-csd-server":
    server_model = create_model(model_name, is_moon=True, out_dim=out_dim, n_classes=n_classes, use_projection_head=True)
    public_data_loader = load_public_data(data_loader_config)

    return FedMdParamsShareCsdServer.create_server(
      server_model=server_model,
      public_data_loader=public_data_loader,
      server_device=server_device,
      use_wandb=use_wandb,
      run_config=context.run_config,
      num_rounds=num_rounds,
    )
  else:
    raise ValueError(f"Unknown server name: {server_name}.")


# Create ServerApp
app = ServerApp(server_fn=server_fn)
