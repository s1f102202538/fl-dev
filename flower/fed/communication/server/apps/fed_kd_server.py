from fed.util.model_util import weighted_average
from flwr.server import ServerAppComponents, ServerConfig

from ..strategy.fed_kd import FedKD


class FedKDServer:
  @staticmethod
  def create_server(use_wandb: bool, run_config, num_rounds: int) -> ServerAppComponents:
    strategy = FedKD(
      run_config=run_config,
      use_wandb=use_wandb,
      fraction_fit=1.0,
      fraction_evaluate=1.0,
      evaluate_metrics_aggregation_fn=weighted_average,
      logit_temperature=4.0,
      entropy_threshold=0.2,
      max_history_rounds=2,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)
