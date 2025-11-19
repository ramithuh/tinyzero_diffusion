"""
Main training script for diffusion language models.

This script uses Hydra for configuration management, loading settings from
configs/config.yaml to orchestrate the complete training pipeline.

To control training configs, look at the following directories:
    + configs/model : specify model architecture and parameters
    + configs/data  : specify data loading and processing parameters
    + configs/training : define accelerator, epochs, logging interval etc.

Finally, configs/config.yaml combines above components (model, data, training, logger)
"""
import os
import hydra
import wandb
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig, OmegaConf
from tzd.utils.generation import log_generations

OmegaConf.register_new_resolver("oc.eval", eval)

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Using the configs defined, it will instantiate the datamodule & model,
    train the model, and log metrics to wandb.

    :param cfg: Hydra configuration object containing all settings
    :type cfg: DictConfig
    :return: None
    """

    # This need to be set to avoid parallelism issues with huggingface tokenizers in pytorch lightning
    if "AutoTokenizer" in cfg.tokenizer._target_:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print(f"Training with config:\n{cfg}")

    L.seed_everything(cfg.seed)

    # Instantiate the datamodule
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()

    # Instantiate the model
    model = hydra.utils.instantiate(cfg.model)
    print(model)

    # Log model information
    print(f"Model: {model.model_alias}, Parameters: {model.get_num_params():,}")

    # Initialize trainer
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=cfg.training.save_dir,
        save_top_k=3,
        filename=f'{model.model_alias}-{{epoch:02d}}-{{val_loss:.2f}}',
        save_last=True,
    )

    trainer = L.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        num_nodes=cfg.training.num_nodes,
        strategy=cfg.training.strategy,
        log_every_n_steps=cfg.training.log_every_n_steps,
        val_check_interval=cfg.training.val_check_interval,
        accumulate_grad_batches=cfg.training.gradient_accumulation_steps,
        enable_progress_bar=True,
        logger=hydra.utils.instantiate(cfg.logger) if cfg.logger else None,
        callbacks=[checkpoint_callback]
    )

    # Log complete Hydra configuration to wandb
    if trainer.logger:
        trainer.logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    # Train the model
    trainer.fit(model, datamodule)

    # Test the model
    print("\nRunning test phase")
    trainer.test(model, datamodule)

    # After training, log generations to wandb
    test_table = wandb.Table(
            columns=["Epoch", "Step", "Temperature", "Sample 1", "Sample 2", "Sample 3"],
            log_mode="MUTABLE"
        )
    log_generations(
        trainer=trainer,
        model=model,
        datamodule=datamodule,
        epoch=trainer.current_epoch,
        step=trainer.global_step,
        temperatures=[0.5, 1.0, 1.5],
        num_samples=3,
        wandb_table=test_table,
        stage="test"
    )

if __name__ == "__main__":
    # Hydra decorator will automatically pass cfg argument
    main()
