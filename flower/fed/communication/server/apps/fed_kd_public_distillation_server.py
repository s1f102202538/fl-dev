from fed.models.base_model import BaseModel
from fed.util.model_util import weighted_average
from flwr.server import ServerAppComponents, ServerConfig
from torch import device
from torch.utils.data import DataLoader

from ..strategy.fed_kd_public_distillation import FedKDPublicDistillation


class FedKDPublicDistillationServer:
  @staticmethod
  def create_server(
    server_model: BaseModel,
    public_data_loader: DataLoader,
    server_device: device,
    use_wandb: bool,
    run_config,
    num_rounds: int,
  ) -> ServerAppComponents:
    """Create FedKD Public Distillation server with public data pre-training.

    Args:
        server_model: Pre-created server-side model
        public_data_loader: Pre-loaded public data loader
        server_device: Device to run server model on
        use_wandb: Whether to use Weights & Biases for logging
        run_config: Configuration for the federated learning run
        num_rounds: Number of federated learning rounds

    Returns:
        ServerAppComponents with FedKDPublicDistillation strategy
    """
    # Move server model to specified device
    server_model.to(server_device)

    # Create strategy with server model and public data
    strategy = FedKDPublicDistillation(
      server_model=server_model,
      public_data_loader=public_data_loader,
      run_config=run_config,
      use_wandb=use_wandb,
      fraction_fit=1.0,
      fraction_evaluate=1.0,
      evaluate_metrics_aggregation_fn=weighted_average,
      distillation_epochs=5,  # Knowledge distillation training epochs (rounds 2+)
      distillation_learning_rate=0.001,  # Knowledge distillation learning rate (rounds 2+)
      public_training_epochs=5,  # Public data training epochs (all rounds)
      public_training_learning_rate=0.01,  # Public data training learning rate (all rounds)
      kd_temperature=3.0,  # Knowledge distillation temperature
      min_fit_clients=5,
      min_evaluate_clients=5,
      min_available_clients=5,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)
