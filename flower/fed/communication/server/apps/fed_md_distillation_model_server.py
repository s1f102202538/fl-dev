from fed.models.base_model import BaseModel
from fed.util.model_util import weighted_average
from flwr.server import ServerAppComponents, ServerConfig
from torch import device
from torch.utils.data import DataLoader

from ..strategy.fed_md_distillation_model import FedMdDistillationModel


class FedMdDistillationModelServer:
  @staticmethod
  def create_server(
    server_model: BaseModel,
    public_data_loader: DataLoader,
    server_device: device,
    use_wandb: bool,
    run_config,
    num_rounds: int,
  ) -> ServerAppComponents:
    """Create FedKD Distillation Model server with server-side model training.

    Args:
        server_model: Pre-created server-side model
        public_data_loader: Pre-loaded public data loader
        server_device: Device to run server model on
        use_wandb: Whether to use Weights & Biases for logging
        run_config: Configuration for the federated learning run
        num_rounds: Number of federated learning rounds

    Returns:
        ServerAppComponents with FedKDDistillationModel strategy
    """
    # Move server model to specified device
    server_model.to(server_device)

    # Create strategy with server model and public data
    strategy = FedMdDistillationModel(
      server_model=server_model,
      public_data_loader=public_data_loader,
      run_config=run_config,
      use_wandb=use_wandb,
      fraction_fit=1.0,
      fraction_evaluate=1.0,
      evaluate_metrics_aggregation_fn=weighted_average,
      server_training_epochs=20,  # Server model training epochs
      server_learning_rate=0.001,  # Server model learning rate
      kd_temperature=3.0,  # Knowledge distillation temperature
      min_fit_clients=5,
      min_evaluate_clients=5,
      min_available_clients=5,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)
