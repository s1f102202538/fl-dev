from typing import Dict

from flower.common.util.util import weighted_average
from flwr.common import Context, Scalar
from flwr.server import ServerAppComponents, ServerConfig

from fl.flower.server.strategy.fed_kd import FedKD


class FedKDServer:
  @staticmethod
  def on_fit_config(server_round: int) -> Dict[str, Scalar]:
    """Construct `config` that clients receive when running `fit()`"""
    lr = 0.1
    # Enable a simple form of learning rate decay
    if server_round > 10:
      lr /= 2
    return {"lr": lr}

  @staticmethod
  def server_fn(context: Context) -> ServerAppComponents:
    # Read from config
    num_rounds = int(context.run_config["num-server-rounds"])
    fraction_fit = context.run_config["fraction-fit"]
    fraction_eval = context.run_config["fraction-evaluate"]

    # Define strategy
    strategy = FedKD(
      run_config=context.run_config,
      use_wandb=bool(context.run_config["use-wandb"]),
      fraction_fit=float(fraction_fit),
      fraction_evaluate=float(fraction_eval),
      on_fit_config_fn=FedKDServer.on_fit_config,
      evaluate_metrics_aggregation_fn=weighted_average,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)
