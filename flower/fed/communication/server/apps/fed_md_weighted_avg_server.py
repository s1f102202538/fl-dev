from fed.util.model_util import weighted_average
from flwr.server import ServerAppComponents, ServerConfig

from ..strategy.fed_md_weighted_avg import FedMdWeightedAvg


class FedMdWeightedAvgServer:
  @staticmethod
  def create_server(use_wandb: bool, run_config, num_rounds: int) -> ServerAppComponents:
    strategy = FedMdWeightedAvg(
      run_config=run_config,
      use_wandb=use_wandb,
      fraction_fit=1.0,
      fraction_evaluate=1.0,
      evaluate_metrics_aggregation_fn=weighted_average,
      max_history_rounds=2,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)
