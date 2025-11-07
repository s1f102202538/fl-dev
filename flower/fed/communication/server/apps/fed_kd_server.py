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
      # Enhanced logit filtering parameters for Non-IID environments
      logit_temperature=3.0,  # ロジット正規化用温度
      kd_temperature=5.0,  # 知識蒸留用温度（最適化済み）
      entropy_threshold=0.1,  # エントロピー閾値（Non-IID対応）
      confidence_threshold=0.3,  # 信頼度閾値
      adaptive_filtering=True,  # 適応的フィルタリング有効
      max_history_rounds=3,  # 履歴保持数を増加
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)
