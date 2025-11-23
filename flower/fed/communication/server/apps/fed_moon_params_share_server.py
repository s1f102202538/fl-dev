from fed.models.base_model import BaseModel
from fed.util.model_util import get_weights, weighted_average
from flwr.common import ndarrays_to_parameters
from flwr.server import ServerAppComponents, ServerConfig
from torch import device
from torch.utils.data import DataLoader

from ..strategy.fed_moon_params_share import FedMoonParamsShare


class FedMoonParamsShareServer:
  @staticmethod
  def create_server(
    server_model: BaseModel,
    public_data_loader: DataLoader,
    server_device: device,
    use_wandb: bool,
    run_config,
    num_rounds: int,
  ) -> ServerAppComponents:
    """Create FedMoon Parameter Sharing server.

    This server:
    1. Sends model parameters to clients
    2. Receives logits from clients after MOON training
    3. Aggregates logits and performs knowledge distillation on server model

    Args:
        server_model: Pre-created server-side model
        public_data_loader: Pre-loaded public data loader for distillation
        server_device: Device to run server model on
        use_wandb: Whether to use Weights & Biases for logging
        run_config: Configuration for the federated learning run
        num_rounds: Number of federated learning rounds

    Returns:
        ServerAppComponents with FedMoonParamsShare strategy
    """
    # Move server model to specified device
    server_model.to(server_device)

    # Get initial parameters from server model
    initial_parameters = ndarrays_to_parameters(get_weights(server_model))

    # Create strategy with logit aggregation and server-side distillation
    strategy = FedMoonParamsShare(
      server_model=server_model,
      public_data_loader=public_data_loader,
      run_config=run_config,
      use_wandb=use_wandb,
      fraction_fit=1.0,
      fraction_evaluate=1.0,
      initial_parameters=initial_parameters,
      evaluate_metrics_aggregation_fn=weighted_average,
      kd_temperature=3.0,  # Knowledge distillation temperature
      min_fit_clients=5,
      min_evaluate_clients=5,
      min_available_clients=5,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)
