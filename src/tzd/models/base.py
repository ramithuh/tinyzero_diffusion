from abc import abstractmethod
from typing import Any, Dict, Union

import torch
import lightning as L

class BaseModel(L.LightningModule):
    """
    Base class for all models in the protein language model project.
    
    :param model_alias: Alias for the model, used for logging and identification.
    :param lr: Learning rate for the optimizer.
    """
    def __init__(self, model_alias: str = "base_model", lr: float = 1e-4):
        super().__init__()
        self.model_alias = model_alias
        self.save_hyperparameters(logger=False)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        pass

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss = self._compute_loss(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss = self._compute_loss(batch)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss = self._compute_loss(batch)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        return loss

    @abstractmethod
    def _compute_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Compute loss for a batch of data."""
        pass

    def configure_optimizers(self) -> Union[torch.optim.Optimizer, Dict[str, Any]]:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        # Override in subclasses to add scheduler:
        # return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return optimizer

    def get_num_params(self) -> int:
        """Get the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
