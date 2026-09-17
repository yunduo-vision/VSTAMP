from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

from .config import save_config
from .data.dataset import ProcessedDataset
from .data.episode_sampler import EpisodeSampler
from .data.partial_views import alignment_target, prefix_retention, warp_tokens
from .evaluation.protocols import fixed_episodes
from .evaluation.recognition import evaluate_episodes
from .models.losses import alignment_loss, assert_finite, episodic_loss, supervised_contrastive_loss
from .models.vstamp import Fingerprint, VSTAMP
from .utils.checkpoint import save_checkpoint
from .utils.logging import CSVLogger, write_json
from .utils.seed import set_seed


def _reshape_fingerprint(fingerprint: Fingerprint, ways: int, shots: int) -> Fingerprint:
    return Fingerprint(
        fingerprint.z.reshape(ways, shots, -1),
        fingerprint.tokens.reshape(ways, shots, *fingerprint.tokens.shape[1:]),
        fingerprint.reliability.reshape(ways, shots, -1),
    )


def _structured_view_loss(
    model: VSTAMP,
    original: Fingerprint,
    features: dict[int, Tensor],
    masks: dict[int, Tensor],
    cfg: Mapping[str, Any],
) -> tuple[Tensor, dict[str, Tensor], Tensor]:
    batch = original.z.shape[0]
    device = original.z.device
    dtype = original.z.dtype
    aug = cfg["augmentation"]
    ablation = cfg.get("ablation", {})
    rho = torch.ones(batch, device=device, dtype=dtype)
    if bool(ablation.get("prefix_transform", True)):
        trigger = torch.rand(batch, device=device) < float(aug["prefix_probability"])
        sampled = torch.empty(batch, device=device, dtype=dtype).uniform_(
            float(aug["prefix_min"]), float(aug["prefix_max"])
        )
        rho = torch.where(trigger, sampled, rho)
    partial_x, partial_m = prefix_retention(features, masks, rho)
    partial = model.encode(partial_x, partial_m)
    eta = torch.zeros(batch, device=device, dtype=dtype)
    if bool(ablation.get("progress_warp", True)):
        trigger = torch.rand(batch, device=device) < float(aug["warp_probability"])
        sampled = torch.empty(batch, device=device, dtype=dtype).uniform_(
            float(aug["warp_eta_min"]), float(aug["warp_eta_max"])
        )
        eta = torch.where(trigger, sampled, eta)
    warped_tokens, warped_reliability, coordinates = warp_tokens(
        partial.tokens, partial.reliability, eta
    )
    transformed = Fingerprint(partial.z, warped_tokens, warped_reliability)
    metrics = model.compare(original, transformed)
    target, valid_columns = alignment_target(
        original.reliability,
        warped_reliability,
        coordinates,
        sigma_seconds=float(cfg["loss"]["alignment_sigma_seconds"]),
        epsilon=float(cfg["numerics"]["epsilon"]),
    )
    loss = alignment_loss(
        metrics.alignment_mass,
        target,
        valid_columns,
        float(cfg["numerics"]["epsilon"]),
    )
    return (
        loss,
        {
            "affinity": metrics.affinity,
            "pma_score": metrics.pma_score,
            "alignment_mass": metrics.alignment_mass,
        },
        partial.z,
    )


def train(cfg: dict[str, Any], seed: int, device: torch.device) -> Path:
    set_seed(seed, deterministic=bool(cfg["numerics"]["deterministic"]))
    dataset_name = str(cfg["dataset"]["name"])
    processed_root = Path(cfg["data"]["processed_root"])
    train_data = ProcessedDataset(processed_root, split="train")
    validation_data = ProcessedDataset(processed_root, split="validation")
    held_out = str(cfg["dataset"].get("held_out_bandwidth", ""))
    train_records = [record for record in train_data.records if not held_out or record.bandwidth != held_out]
    validation_records = [
        record for record in validation_data.records if not held_out or record.bandwidth != held_out
    ]
    sampler = EpisodeSampler(train_records, seed)
    output = (
        Path(cfg["output"]["root"])
        / dataset_name
        / str(cfg["output"]["experiment_name"])
        / f"seed_{seed}"
    ).resolve()
    output.mkdir(parents=True, exist_ok=True)
    save_config(cfg, output / "config.yaml")
    episode_stream = (output / "train_episodes.jsonl").open("w", encoding="utf-8")

    model = VSTAMP(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    total_episodes = int(cfg["training"]["episodes"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_episodes,
        eta_min=float(cfg["training"]["minimum_learning_rate"]),
    )
    mixed = bool(cfg["training"]["mixed_precision"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=mixed)
    logger = CSVLogger(
        output / "training_log.csv",
        [
            "episode",
            "total_loss",
            "episodic_loss",
            "alignment_loss",
            "supcon_loss",
            "learning_rate",
            "validation_accuracy",
        ],
    )
    validation_protocol = str(cfg["evaluation"]["validation_protocol"])
    validation_count = min(300, int(cfg["evaluation"]["fixed_episodes"]))
    validation_episodes = fixed_episodes(
        validation_records,
        validation_protocol,
        validation_count,
        seed=10000 + seed,
        dataset_name=dataset_name,
        cache_dir=processed_root / "episodes",
    )
    best_accuracy = float("-inf")
    ways = int(cfg["training"]["n_way"])
    shots = int(cfg["training"]["n_shot"])
    queries_per_class = int(cfg["training"]["n_query"])
    try:
        for step in range(1, total_episodes + 1):
            model.train()
            episode = sampler.sample(ways, shots, queries_per_class)
            episode_stream.write(json.dumps(episode.__dict__) + "\n")
            support_ids = [session for group in episode.support_ids for session in group]
            query_ids = [session for group in episode.query_ids for session in group]
            all_ids = support_ids + query_ids
            features, masks = train_data.tensors(all_ids, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=mixed):
                episodic_features, episodic_masks = features, masks
                if (
                    model.global_only or model.operator in {"diagonal", "soft_dtw"}
                ) and bool(cfg.get("ablation", {}).get("prefix_transform", True)):
                    batch_size = len(all_ids)
                    trigger = torch.rand(batch_size, device=device) < float(
                        cfg["augmentation"]["prefix_probability"]
                    )
                    sampled = torch.empty(batch_size, device=device).uniform_(
                        float(cfg["augmentation"]["prefix_min"]),
                        float(cfg["augmentation"]["prefix_max"]),
                    )
                    rho = torch.where(trigger, sampled, torch.ones_like(sampled))
                    episodic_features, episodic_masks = prefix_retention(
                        features, masks, rho
                    )
                fingerprint = model.encode(episodic_features, episodic_masks)
                support_flat = Fingerprint(
                    fingerprint.z[: len(support_ids)],
                    fingerprint.tokens[: len(support_ids)],
                    fingerprint.reliability[: len(support_ids)],
                )
                support = _reshape_fingerprint(support_flat, ways, shots)
                query = Fingerprint(
                    fingerprint.z[len(support_ids) :],
                    fingerprint.tokens[len(support_ids) :],
                    fingerprint.reliability[len(support_ids) :],
                )
                candidate_scores, _, pair_metrics = model.score_candidates(query, support)
                targets = torch.arange(ways, device=device).repeat_interleave(queries_per_class)
                epi = episodic_loss(
                    candidate_scores, targets, float(cfg["matching"]["episode_temperature"])
                )
                labels = torch.cat(
                    (
                        torch.arange(ways, device=device).repeat_interleave(shots),
                        torch.arange(ways, device=device).repeat_interleave(queries_per_class),
                    )
                )
                supcon_features = fingerprint.z
                supcon_labels = labels
                if model.operator == "pma" and not model.global_only:
                    ali, alignment_tensors, view_z = _structured_view_loss(
                        model, fingerprint, features, masks, cfg
                    )
                    supcon_features = torch.cat((fingerprint.z, view_z), dim=0)
                    supcon_labels = torch.cat((labels, labels), dim=0)
                else:
                    ali = fingerprint.z.sum() * 0.0
                    alignment_tensors = {
                        "affinity": pair_metrics.affinity,
                        "pma_score": pair_metrics.pma_score,
                        "alignment_mass": pair_metrics.alignment_mass,
                    }
                supcon = supervised_contrastive_loss(
                    supcon_features,
                    supcon_labels,
                    float(cfg["loss"]["supcon_temperature"]),
                )
                loss = (
                    epi
                    + float(cfg["loss"]["lambda_alignment"]) * ali
                    + float(cfg["loss"]["lambda_supcon"]) * supcon
                )
            assert_finite("loss", loss, f"episode={step} ids={all_ids[:4]}")
            assert_finite("candidate score", candidate_scores, f"episode={step}")
            for name, value in alignment_tensors.items():
                assert_finite(name, value, f"episode={step}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(
                model.parameters(),
                float(cfg["training"]["gradient_clip"]),
                error_if_nonfinite=True,
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            validation_accuracy: float | str = ""
            if (
                step % int(cfg["training"]["validation_interval"]) == 0
                or step == total_episodes
            ):
                metrics, _ = evaluate_episodes(model, validation_data, validation_episodes, device)
                validation_accuracy = float(metrics["accuracy"])
                if validation_accuracy > best_accuracy:
                    best_accuracy = validation_accuracy
                    save_checkpoint(
                        output / "best.pt",
                        model=model.state_dict(),
                        optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(),
                        episode=step,
                        validation_accuracy=best_accuracy,
                        seed=seed,
                        config=cfg,
                    )
            if (
                step % int(cfg["training"]["checkpoint_interval"]) == 0
                or step == total_episodes
            ):
                save_checkpoint(
                    output / "last.pt",
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),
                    episode=step,
                    validation_accuracy=best_accuracy,
                    seed=seed,
                    config=cfg,
                )
            if step % int(cfg["training"]["log_interval"]) == 0 or validation_accuracy != "":
                logger.log(
                    {
                        "episode": step,
                        "total_loss": float(loss.detach().cpu()),
                        "episodic_loss": float(epi.detach().cpu()),
                        "alignment_loss": float(ali.detach().cpu()),
                        "supcon_loss": float(supcon.detach().cpu()),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "validation_accuracy": validation_accuracy,
                    }
                )
    finally:
        episode_stream.close()
    write_json(output / "metrics.json", {"best_validation_accuracy": best_accuracy, "seed": seed})
    return output
