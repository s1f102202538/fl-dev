from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union, override

from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, MetricsAggregationFn, NDArrays, Parameters, Scalar
from flwr.common.logger import log
from flwr.common.typing import UserConfig
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
from flwr.server.strategy.aggregate import weighted_loss_avg
from torch import Tensor, stack, tensor

from fl.flower.common.util.util import base64_to_tensor, create_run_dir, tensor_to_base64


class FedKD(Strategy):
  """Federated Knowledge Distillation (FedKD) strategy."""

  def __init__(
    self,
    *,
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 1.0,
    min_fit_clients: int = 2,
    min_evaluate_clients: int = 2,
    min_available_clients: int = 2,
    evaluate_fn: Optional[
      Callable[
        [int, NDArrays, dict[str, Scalar]],
        Optional[tuple[float, dict[str, Scalar]]],
      ]
    ] = None,
    on_fit_config_fn: Optional[Callable[[int], dict[str, Scalar]]] = None,
    on_evaluate_config_fn: Optional[Callable[[int], dict[str, Scalar]]] = None,
    accept_failures: bool = True,
    initial_parameters: Optional[Parameters] = None,
    fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
    evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
    inplace: bool = True,
    run_config: UserConfig,
    use_wandb: bool = False,
  ) -> None:
    self.fraction_fit = fraction_fit
    self.fraction_evaluate = fraction_evaluate
    self.min_fit_clients = min_fit_clients
    self.min_evaluate_clients = min_evaluate_clients
    self.min_available_clients = min_available_clients
    self.evaluate_fn = evaluate_fn
    self.on_fit_config_fn = on_fit_config_fn
    self.on_evaluate_config_fn = on_evaluate_config_fn
    self.accept_failures = accept_failures
    self.initial_parameters = initial_parameters
    self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
    self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
    self.inplace = inplace
    self.avg_logits: List[Tensor] = []  # 集約したロジット

    self.save_path, self.run_dir = create_run_dir(run_config)
    self.use_wandb = use_wandb

  @override
  def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
    """Initialize the (global) model parameters.

    Parameters
    ----------
    client_manager : ClientManager
        The client manager which holds all currently connected clients.

    Returns
    -------
    parameters : Optional[Parameters]
        If parameters are returned, then the server will treat these as the
        initial global model parameters.
    """

    # サーバはモデルを持たないためモデルパラメータは None を返す
    return None

  @override
  def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy, FitIns]]:
    """Configure the next round of training.

    Parameters
    ----------
    server_round : int
        The current round of federated learning.
    parameters : Parameters
        The current (global) model parameters.
    client_manager : ClientManager
        The client manager which holds all currently connected clients.

    Returns
    -------
    fit_configuration : List[Tuple[ClientProxy, FitIns]]
        A list of tuples. Each tuple in the list identifies a `ClientProxy` and the
        `FitIns` for this particular `ClientProxy`. If a particular `ClientProxy`
        is not included in this list, it means that this `ClientProxy`
        will not participate in the next round of federated learning.
    """

    config = self.on_fit_config_fn(server_round) if self.on_fit_config_fn else {}
    if self.avg_logits:
      config["avg_logits"] = tensor_to_base64(self.avg_logits)
    # parameters は None を返す
    fit_ins = FitIns(parameters, config)

    # 有効になっているクライアントの取得
    sample_size = int(self.fraction_fit * client_manager.num_available())
    clients = client_manager.sample(num_clients=sample_size, min_num_clients=self.min_fit_clients)

    # Return client/config pairs
    return [(client, fit_ins) for client in clients]

  @override
  def aggregate_fit(
    self,
    server_round: int,
    results: List[Tuple[ClientProxy, FitRes]],
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
  ) -> Tuple[Optional[Parameters], dict[str, Scalar]]:
    """Aggregate training results.

    Parameters
    ----------
    server_round : int
        The current round of federated learning.
    results : List[Tuple[ClientProxy, FitRes]]
        Successful updates from the previously selected and configured
        clients. Each pair of `(ClientProxy, FitRes)` constitutes a
        successful update from one of the previously selected clients. Not
        that not all previously selected clients are necessarily included in
        this list: a client might drop out and not submit a result. For each
        client that did not submit an update, there should be an `Exception`
        in `failures`.
    failures : List[Union[Tuple[ClientProxy, FitRes], BaseException]]
        Exceptions that occurred while the server was waiting for client
        updates.

    Returns
    -------
    parameters : Tuple[Optional[Parameters], Dict[str, Scalar]]
        If parameters are returned, then the server will treat these as the
        new global model parameters (i.e., it will replace the previous
        parameters with the ones returned from this method). If `None` is
        returned (e.g., because there were only failures and no viable
        results) then the server will no update the previous model
        parameters, the updates received in this round are discarded, and
        the global model parameters remain the same.
    """

    logits_list = []
    for _, fit_res in results:
      if "logits" in fit_res.metrics:
        logits_tensor = base64_to_tensor(str(fit_res.metrics["logits"]))
        logits_list.append(logits_tensor)

    if logits_list:
      avg_logits = stack(logits_list).mean(dim=0)
      # 集約したロジットの更新
      self.avg_logits = avg_logits.tolist()

    return super().aggregate_fit(server_round, results, failures)

  @override
  def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy, EvaluateIns]]:
    """Configure the next round of evaluation.

    Parameters
    ----------
    server_round : int
        The current round of federated learning.
    parameters : Parameters
        The current (global) model parameters.
    client_manager : ClientManager
        The client manager which holds all currently connected clients.

    Returns
    -------
    evaluate_configuration : List[Tuple[ClientProxy, EvaluateIns]]
        A list of tuples. Each tuple in the list identifies a `ClientProxy` and the
        `EvaluateIns` for this particular `ClientProxy`. If a particular
        `ClientProxy` is not included in this list, it means that this
        `ClientProxy` will not participate in the next round of federated
        evaluation.
    """

    # 評価用の設定を作成
    config = {}
    if self.avg_logits:
      config["avg_logits"] = tensor_to_base64(self.avg_logits)
    evaluate_ins = EvaluateIns(parameters, config)

    # 評価に参加するクライアントをサンプリング
    sample_size = int(self.fraction_evaluate * client_manager.num_available())
    clients = client_manager.sample(num_clients=sample_size, min_num_clients=self.min_evaluate_clients)

    # Return client/config pairs
    return [(client, evaluate_ins) for client in clients]

  @override
  def aggregate_evaluate(
    self,
    server_round: int,
    results: List[Tuple[ClientProxy, EvaluateRes]],
    failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
  ) -> Tuple[Optional[float], Dict[str, Scalar]]:
    """Aggregate evaluation results.

    Parameters
    ----------
    server_round : int
        The current round of federated learning.
    results : List[Tuple[ClientProxy, FitRes]]
        Successful updates from the
        previously selected and configured clients. Each pair of
        `(ClientProxy, FitRes` constitutes a successful update from one of the
        previously selected clients. Not that not all previously selected
        clients are necessarily included in this list: a client might drop out
        and not submit a result. For each client that did not submit an update,
        there should be an `Exception` in `failures`.
    failures : List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]
        Exceptions that occurred while the server was waiting for client updates.

    Returns
    -------
    aggregation_result : Tuple[Optional[float], Dict[str, Scalar]]
        The aggregated evaluation result. Aggregation typically uses some variant
        of a weighted average.
    """

    if not results:
      return None, {}
    # Do not aggregate if there are failures and failures are not accepted
    if not self.accept_failures and failures:
      return None, {}

    # Aggregate loss
    loss_aggregated = weighted_loss_avg([(evaluate_res.num_examples, evaluate_res.loss) for _, evaluate_res in results])

    # Aggregate custom metrics if aggregation fn was provided
    metrics_aggregated = {}
    if self.evaluate_metrics_aggregation_fn:
      eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
      metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
    elif server_round == 1:  # Only log this warning once
      log(WARNING, "No evaluate_metrics_aggregation_fn provided")

    return loss_aggregated, metrics_aggregated

  @override
  def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
    """Evaluate the current model parameters.

    This function can be used to perform centralized (i.e., server-side) evaluation
    of model parameters.

    Parameters
    ----------
    server_round : int
        The current round of federated learning.
    parameters: Parameters
        The current (global) model parameters.

    Returns
    -------
    evaluation_result : Optional[Tuple[float, Dict[str, Scalar]]]
        The evaluation result, usually a Tuple containing loss and a
        dictionary containing task-specific metrics (e.g., accuracy).
    """

    # サーバはモデルを持たないため評価は行わない
