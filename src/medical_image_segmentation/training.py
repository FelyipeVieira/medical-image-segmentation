"""PyTorch Lightning training module and trainer construction."""

from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger
from torch import Tensor, nn


class SegmentationTask(L.LightningModule):
    """Model-agnostic Lightning segmentation task.

    Example:
        >>> task = SegmentationTask(model, loss, learning_rate=1e-4)  # doctest: +SKIP
    """

    def __init__(
        self, model: nn.Module, loss: nn.Module, learning_rate: float = 1e-4
    ) -> None:
        """Register the network, objective, and optimizer settings."""
        super().__init__()
        self.model, self.loss, self.learning_rate = model, loss, learning_rate
        self.save_hyperparameters(ignore=("model", "loss"))

    def forward(self, image: Tensor) -> Tensor:
        """Return raw segmentation logits."""
        return self.model(image)

    def _step(self, batch: dict[str, Tensor], phase: str) -> Tensor:
        logits = self(batch["image"])
        target = batch["mask"]
        if logits.shape[1] > 1 and target.ndim == logits.ndim - 1:
            target = (
                torch.nn.functional.one_hot(target, logits.shape[1])
                .movedim(-1, 1)
                .float()
            )
        elif target.ndim == logits.ndim - 1:
            target = target.unsqueeze(1).float()
        loss = self.loss(logits, target)
        self.log(f"{phase}_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def training_step(self, batch: dict[str, Tensor], batch_index: int) -> Tensor:
        """Compute and log one training loss."""
        return self._step(batch, "train")

    def validation_step(self, batch: dict[str, Tensor], batch_index: int) -> Tensor:
        """Compute and log one validation loss."""
        return self._step(batch, "val")

    def configure_optimizers(self) -> dict[str, object]:
        """Configure AdamW and plateau-based learning-rate scheduling."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }


def create_trainer(
    output: str | Path, epochs: int, precision: str = "16-mixed"
) -> L.Trainer:
    """Create a trainer with AMP, checkpoints, early stopping, and TensorBoard."""
    output = Path(output)
    callbacks = [
        ModelCheckpoint(
            dirpath=output / "checkpoints",
            monitor="val_loss",
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(monitor="val_loss", patience=15),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    return L.Trainer(
        max_epochs=epochs,
        precision=precision,
        callbacks=callbacks,
        logger=TensorBoardLogger(output, name="tensorboard"),
        default_root_dir=output,
    )
