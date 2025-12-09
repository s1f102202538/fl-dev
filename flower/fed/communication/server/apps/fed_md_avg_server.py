from fed.util.model_util import weighted_average
from flwr.server import ServerAppComponents, ServerConfig

from ..strategy.fed_md_avg import FedMdAvg


class FedMdAvgServer:
  @staticmethod
  def create_server(use_wandb: bool, run_config, num_rounds: int) -> ServerAppComponents:
    strategy = FedMdAvg(
      run_config=run_config,
      use_wandb=use_wandb,
      fraction_fit=1.0,
      fraction_evaluate=1.0,
      evaluate_metrics_aggregation_fn=weighted_average,
      kd_temperature=3.0,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)
