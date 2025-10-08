from flwr.common import Context
from flwr.server import ServerAppComponents, ServerConfig

from fl.flower.common.util.model_util import weighted_average
from flower.server.strategy.fed_kd import FedKD


class FedKDServer:
  @staticmethod
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
      evaluate_metrics_aggregation_fn=weighted_average,
      logit_temperature=4.0,  # 初期温度
      enable_adaptive_temperature=True,
      entropy_threshold=0.2,  # 品質閾値で多くのロジットを活用
      max_history_rounds=2,  # 履歴数
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)
