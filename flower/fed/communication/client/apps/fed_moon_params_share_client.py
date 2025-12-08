import copy
from typing import Dict, Tuple

import torch
from fed.algorithms.moon import MoonContrastiveLearning, MoonTrainer
from fed.models.base_model import BaseModel
from fed.task.cnn_task import CNNTask
from fed.util.model_util import (
  batch_list_to_base64,
  load_model_from_state,
  save_model_to_state,
  set_weights,
)
from flwr.client import NumPyClient
from flwr.common import RecordDict
from flwr.common.typing import NDArrays, UserConfigValue
from torch.utils.data import DataLoader


class FedMoonParamsShareClient(NumPyClient):
  def __init__(
    self,
    net: BaseModel,
    client_state: RecordDict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    public_test_data: DataLoader,
    local_epochs: UserConfigValue,
  ) -> None:
    super().__init__()
    self.net = net
    self.client_state = client_state
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.local_epochs: int = int(local_epochs)
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.net.to(self.device)
    self.public_test_data = public_test_data

    self.virtual_global_model = net

    # Model state storage keys
    self.local_model_name = "local-model"
    self.global_model_name = "global-model"

    # Initialize Moon contrastive learning with optimized parameters
    self.moon_learner = MoonContrastiveLearning(
      mu=3.0,
      temperature=0.3,
      device=self.device,
    )

    # Initialize Moon trainer
    self.moon_trainer = MoonTrainer(
      moon_learner=self.moon_learner,
      device=self.device,
    )

  def fit(self, parameters: NDArrays, config: Dict) -> Tuple[NDArrays, int, Dict]:
    # Apply server parameters to local model
    if parameters is not None and len(parameters) > 0:
      print("[INFO] Applying server parameters to local model")
      set_weights(self.net, parameters)
    else:
      print("[INFO] No server parameters provided, using current model state")

    # Load previous round model if available
    previous_round_model = self._load_previous_round_model()

    # Update model history for MOON if not first round
    if previous_round_model is not None:
      self._update_model_history(previous_round_model)

    # Perform training (normal or MOON based on available history)
    train_loss = self._perform_training()

    # Save current model state for next round
    save_model_to_state(self.net, self.client_state, self.local_model_name)

    print(f"Client training loss: {train_loss:.4f}")

    # Generate logits from trained model
    logits = self._generate_logits()
    logits_base64 = batch_list_to_base64(logits)
    print(f"[INFO] Generated {len(logits)} logit batches")

    # Return empty parameters (we're sending logits instead)
    return (
      [],  # Return empty list instead of None
      len(self.train_loader.dataset),  # type: ignore
      {
        "train_loss": train_loss,
        "logits": logits_base64,  # Send logits to server
      },
    )

  def _load_previous_round_model(self) -> BaseModel | None:
    """Load the model from the previous training round."""
    previous_round_model = load_model_from_state(self.client_state, self.net, self.local_model_name)
    if previous_round_model is not None:
      print("[DEBUG] Previous round model loaded successfully")
    else:
      print("[DEBUG] No previous model found, using initial model")
    return previous_round_model

  def _generate_logits(self) -> list:
    """Generate logits from the trained model."""
    logits = CNNTask.inference_with_loca_extended(self.net, self.public_test_data, device=self.device)
    print(f"[DEBUG] Generated logits from {len(logits)} batches")
    return logits

  def _update_model_history(self, previous_round_model: BaseModel) -> None:
    """Update model history for MOON contrastive learning."""
    # Use current model as global model for MOON
    global_model = copy.deepcopy(self.net)

    # MOON対比学習の設定
    self.moon_learner.update_models(previous_round_model, global_model)
    print("Updated Moon learner with 1 previous model and current global model")

  def _perform_training(self) -> float:
    """Perform training using normal or MOON approach based on available model history."""
    # MOON学習が可能かチェック
    if self.moon_learner.previous_model is not None and self.moon_learner.global_model is not None:
      print("[INFO] Performing MOON training with previous model")
      return self.moon_trainer.train_with_moon(
        model=self.net,
        train_loader=self.train_loader,
        lr=0.01,
        epochs=self.local_epochs,
        args_optimizer="sgd",
      )
    else:
      print("[INFO] No previous model available, performing normal training")
      return CNNTask.train(
        net=self.net,
        train_loader=self.train_loader,
        epochs=self.local_epochs,
        lr=0.01,
        device=self.device,
      )

  def evaluate(self, parameters: NDArrays, config: Dict) -> Tuple[float, int, Dict]:
    """Evaluate model performance using server-provided parameters."""
    # parametersがNoneまたは空でない場合、サーバーモデルのパラメータを適用
    if parameters is not None and len(parameters) > 0:
      print("[DEBUG] Applying server model parameters for evaluation")
      set_weights(self.net, parameters)

    loss, accuracy = CNNTask.test(self.net, self.val_loader, self.device)
    return loss, len(self.val_loader.dataset), {"accuracy": accuracy}  # type: ignore
