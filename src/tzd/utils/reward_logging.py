"""Utilities for computing and logging rewards during training."""
import torch
from typing import List, Dict, Any, Optional
from tzd.rl.rewards import countdown_reward_batch


def compute_reward_metrics(
    completions: List[str],
    targets: List[int],
    numbers: List[List[int]]
) -> Dict[str, float]:
    """
    Compute reward metrics for a batch of completions.

    Args:
        completions: List of generated text (with <answer> tags)
        targets: List of target numbers
        numbers: List of available numbers for each problem

    Returns:
        Dictionary of metrics:
        - reward_mean: Average reward (0.0 to 1.0)
        - reward_std: Standard deviation of rewards
        - accuracy: Fraction with perfect score (1.0)
        - format_rate: Fraction with at least format score (>=0.1)
        - failure_rate: Fraction with zero score (0.0)
    """
    if not completions:
        return {}

    scores = countdown_reward_batch(completions, targets, numbers)
    scores_tensor = torch.tensor(scores, dtype=torch.float32)

    metrics = {
        "reward_mean": scores_tensor.mean().item(),
        "reward_std": scores_tensor.std().item() if len(scores) > 1 else 0.0,
        "accuracy": (scores_tensor == 1.0).float().mean().item(),
        "format_rate": (scores_tensor >= 0.1).float().mean().item(),
        "failure_rate": (scores_tensor == 0.0).float().mean().item(),
    }

    return metrics


def log_reward_metrics(
    metrics: Dict[str, float],
    epoch: int,
    step: int,
    stage: str,
    logger: Any
):
    """
    Log reward metrics to WandB.

    Args:
        metrics: Dictionary from compute_reward_metrics
        epoch: Current epoch
        step: Current step
        stage: E.g., "sft_val", "rl_train"
        logger: Trainer logger (WandB)
    """
    if logger is None:
        return

    log_dict = {
        f"{stage}/reward_mean": metrics.get("reward_mean", 0.0),
        f"{stage}/reward_std": metrics.get("reward_std", 0.0),
        f"{stage}/accuracy": metrics.get("accuracy", 0.0),
        f"{stage}/format_rate": metrics.get("format_rate", 0.0),
        f"{stage}/failure_rate": metrics.get("failure_rate", 0.0),
        "epoch": epoch,
        "step": step,
    }

    # Log to experiment (WandB)
    if hasattr(logger, "experiment"):
        logger.experiment.log(log_dict)
    else:
        # Fallback for other loggers
        logger.log_metrics(log_dict)

    # Print summary to console
    print(f"\n{stage.upper()} Rewards @ Epoch {epoch}:")
    print(f"  Accuracy:     {metrics.get('accuracy', 0.0):.1%}")
    print(f"  Format Rate:  {metrics.get('format_rate', 0.0):.1%}")
    print(f"  Mean Reward:  {metrics.get('reward_mean', 0.0):.3f}")
