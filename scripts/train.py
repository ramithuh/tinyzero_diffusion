import hydra
from omegaconf import DictConfig
import torch


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def train(cfg: DictConfig) -> None:
    """
    Main training script powered by Hydra and PyTorch Lightning.
    """
    # Enable TF32 for better performance on Ampere+ GPUs (RTX 30xx, 40xx, etc.)
    torch.set_float32_matmul_precision('high')

    model = hydra.utils.instantiate(cfg.model)
    trainer = hydra.utils.instantiate(cfg.trainer)
    data_module = hydra.utils.instantiate(cfg.data)

    trainer.fit(model, data_module)

if __name__ == "__main__":
    train()
