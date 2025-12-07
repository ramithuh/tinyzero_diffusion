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

    # Initialize callbacks from config or use defaults
    callbacks = []
    if cfg.get("callbacks"):
        for cb_name, cb_cfg in cfg.callbacks.items():
            if cb_cfg is not None:
                cb = hydra.utils.instantiate(cb_cfg, dirpath=cfg.training.save_dir)
                callbacks.append(cb)
    else:
        # Default checkpoint callback for backward compatibility
        checkpoint_callback = ModelCheckpoint(
            monitor='val/reward_mean',
            mode='max',
            dirpath=cfg.training.save_dir,
            save_top_k=3,
            filename=f'{model.model_alias}-{{epoch:02d}}-{{val/reward_mean:.2f}}',
            save_last=True,
        )
        callbacks.append(checkpoint_callback)

    trainer = L.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.get("devices", "auto"),
        num_nodes=cfg.training.get("num_nodes", 1),
        strategy=cfg.training.get("strategy", "auto"),
        log_every_n_steps=cfg.training.get("log_every_n_steps", 50),
        val_check_interval=cfg.training.get("val_check_interval", 1.0),
        accumulate_grad_batches=cfg.training.get("gradient_accumulation_steps", 1),
        enable_progress_bar=True,
        logger=hydra.utils.instantiate(cfg.logger) if cfg.logger else None,
        callbacks=callbacks
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
