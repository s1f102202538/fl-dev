from typing import Dict

from flower.common.util.util import weighted_average
from flower.server.strategy.fed_kd import FedKD
from flwr.common import Context, Scalar
from flwr.server import ServerAppComponents, ServerConfig


class FedKDServer:
  @staticmethod
  def on_fit_config(server_round: int) -> Dict[str, Scalar]:
    """Construct `config` that clients receive when running `fit()`"""
    # より段階的な学習率スケジューリング
    if server_round <= 2:
      lr = 0.02  # 初期は高めの学習率でモデルの表現能力を活用
    elif server_round <= 5:
      lr = 0.01  # 中期は標準的な学習率
    elif server_round <= 8:
      lr = 0.005  # 後期は細かな調整
    else:
      lr = 0.002  # 最終期は非常に細かな調整
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
      logit_temperature=4.0,  # 初期温度
      enable_adaptive_temperature=True,
      entropy_threshold=0.2,  # 品質閾値で多くのロジットを活用
      max_history_rounds=2,  # 履歴数
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)
