"""FedMoon with Logit Sharing: Flower / PyTorch app"""

import copy
from typing import Dict, Tuple

import torch
from fed.algorithms.distillation import Distillation
from fed.algorithms.moon import MoonContrastiveLearning, MoonTrainer
from fed.models.base_model import BaseModel
from fed.task.cnn_task import CNNTask
from fed.util.model_util import (
  base64_to_batch_list,
  batch_list_to_base64,
  filter_and_calibrate_logits,
  load_model_from_state,
  save_model_to_state,
  set_weights,
)
from flwr.client import NumPyClient
from flwr.common import RecordDict
from flwr.common.typing import NDArrays, UserConfigValue
from torch.utils.data import DataLoader


class FedMoonClient(NumPyClient):
  """FedMoon client with logit sharing capabilities."""

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
      mu=1.0,  # 1.0 の方がよい可能性
      temperature=0.5,  # Optimized from analysis: best performance at temp=0.5
      device=self.device,
    )

    # Initialize Moon trainer
    self.moon_trainer = MoonTrainer(
      moon_learner=self.moon_learner,
      device=self.device,
    )

  def fit(self, parameters: NDArrays, config: Dict) -> Tuple[NDArrays, int, Dict]:
    """FedMoon client training with logit sharing and contrastive learning."""
    temperature = float(config.get("temperature", 3.0))

    # Load previous round model if available
    previous_round_model = self._load_previous_round_model()

    # Perform knowledge distillation if logits are available
    if "avg_logits" in config and config["avg_logits"] is not None:
      self._perform_knowledge_distillation(config["avg_logits"], temperature)

    # Update model history for MOON if not first round
    if previous_round_model is not None:
      self._update_model_history(previous_round_model)

    # Perform training (normal or MOON based on available history)
    train_loss = self._perform_training()

    # Save current model state for next round
    save_model_to_state(self.net, self.client_state, self.local_model_name)

    filtered_logits = self._generate_and_filter_logits()

    print(f"Client training loss: {train_loss:.4f}")

    return (
      [],  # Empty list for logit-only sharing (no parameter aggregation)
      len(self.train_loader.dataset),  # type: ignore
      {
        "train_loss": train_loss,
        "logits": batch_list_to_base64(filtered_logits),
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

  def _perform_knowledge_distillation(self, avg_logits: str, temperature: float) -> None:
    """Perform knowledge distillation to create virtual global model."""

    # Use saved global model for distillation if available
    global_model_for_distillation = load_model_from_state(self.client_state, self.net, self.global_model_name)
    if global_model_for_distillation is not None:
      distillation_base_model = global_model_for_distillation
      print("[DEBUG] Using saved global model for knowledge distillation")
    else:
      distillation_base_model = copy.deepcopy(self.net)
      print("[DEBUG] No saved global model found, using current model for distillation")

    logits = base64_to_batch_list(avg_logits)

    # Create virtual global model through distillation
    distillation = Distillation(
      studentModel=distillation_base_model,
      public_data=self.public_test_data,
      soft_targets=logits,
    )

    # Train virtual global model with optimized FedKD parameters
    self.virtual_global_model = distillation.train_knowledge_distillation(
      epochs=5,  # Increased from 3 for better distillation
      learning_rate=0.001,  # Reduced from 0.01 for more stable training
      T=temperature,
      alpha=0.3,  # FedKD paper: KL distillation loss weight
      beta=0.7,  # FedKD paper: CE loss weight
      device=self.device,
    )

    # Use virtual global model as starting point for MOON learning
    self.net = self.virtual_global_model

    # Save distilled model as global model
    save_model_to_state(self.virtual_global_model, self.client_state, self.global_model_name)
    print("[DEBUG] Distilled model saved as global model")

  def _update_model_history(self, previous_round_model: BaseModel) -> None:
    """Update model history for MOON contrastive learning."""
    if self.virtual_global_model is None:
      self.virtual_global_model = self.net

    # MOON対比学習の設定
    self.moon_learner.update_models(previous_round_model, self.virtual_global_model)
    print("Updated Moon learner with 1 previous model and virtual global model")

  def _perform_training(self) -> float:
    """Perform training using normal or MOON approach based on available model history."""
    # MOON学習が可能かチェック
    if self.moon_learner.previous_model is not None and self.moon_learner.global_model is not None:
      print("[INFO] Performing MOON training with previous model")
      return self.moon_trainer.train_with_moon(
        model=self.net,
        train_loader=self.train_loader,
        lr=0.01,  # Optimized from analysis: reduced from 0.01 for better convergence
        epochs=self.local_epochs,
        args_optimizer="sgd",  # Original paper settings
        weight_decay=1e-5,  # Original paper settings
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

  def _generate_and_filter_logits(self) -> list:
    """Generate and calibrate logits for sharing with server without quality filtering."""

    raw_logits = CNNTask.inference(self.net, self.public_test_data, device=self.device)
    print(f"[DEBUG] Raw logits generated: {len(raw_logits)} batches")

    # Apply basic calibration without quality filtering
    filtered_logits = filter_and_calibrate_logits(raw_logits)
    print(f"[DEBUG] Calibrated logits: {len(filtered_logits)} batches (no filtering)")

    return filtered_logits

  # def evaluate(self, parameters: NDArrays, config: Dict) -> Tuple[float, int, Dict]:
  #   """Evaluate model performance with performance tracking."""
  #   # Load the trained model
  #   loaded_model = load_model_from_state(self.client_state, self.net, self.local_model_name)
  #   if loaded_model is not None:
  #     self.net = loaded_model
  #     print("[DEBUG] Model loaded successfully for evaluation")
  #   else:
  #     print("[Warning] No saved model state found, using initial model")

  #   loss, accuracy = CNNTask.test(self.net, self.val_loader, device=self.device)

  #   return (
  #     loss,
  #     len(self.val_loader.dataset),  # type: ignore
  #     {"accuracy": accuracy},
  #   )

  def evaluate(self, parameters: NDArrays, config: Dict) -> Tuple[float, int, Dict]:
    """Evaluate model performance using server-provided parameters."""
    # parametersがNoneまたは空でない場合、サーバーモデルのパラメータを適用
    if parameters is not None and len(parameters) > 0:
      print("[DEBUG] Applying server model parameters for evaluation")
      set_weights(self.net, parameters)

    loss, accuracy = CNNTask.test(self.net, self.val_loader, self.device)
    return loss, len(self.val_loader.dataset), {"accuracy": accuracy}  # type: ignore
