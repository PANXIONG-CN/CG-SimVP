"""Train the direct-field backbone and its climatology / CBAM / geometry variants."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import numpy as np
import os
import json
import random
from tqdm import tqdm
import argparse
from dataloader_clim_aux import TEC_Dataset
from simvp import SimVP
from solar_zenith import configure_geometry

def _seed_worker(worker_id: int):
    worker_seed = (seed + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# 固定随机种子
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# CUDA确定性配置
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def train_simvp_model(config: dict | None = None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    config = config or {}

    # 基础训练参数
    batch_size = int(config.get("batch_size", 4))
    lr = float(config.get("lr", 6e-4))
    epochs = int(config.get("epochs", 300))
    patience = int(config.get("patience", 50))
    val_metric = str(config.get("val_metric", "global")).lower()
    if val_metric not in {"batch", "global"}:
        raise ValueError("val_metric must be batch or global")
    enforce_nonneg = bool(config.get("enforce_nonneg", False))

    # 路径配置
    memmap_root = config.get("data_root", "data")
    model_save_dir = config.get("model_save_dir", "models")

    # 时序与采样参数
    time_step = int(config.get("time_step", 24))
    predict_step = int(config.get("predict_step", 24))
    frames_per_day = int(config.get("frames_per_day", 96))
    use_aux_sza = bool(config.get("use_aux_sza", False))
    use_aux_qd = bool(config.get("use_aux_qd", False))
    geometry_provenance = None
    if use_aux_sza or use_aux_qd:
        grid_npz = config.get("grid_npz") or os.path.join(memmap_root, "grid.npz")
        geometry_cache = config.get("geometry_cache") or os.path.join(memmap_root, "geometry")
        geometry_provenance = configure_geometry(
            grid_npz, geometry_cache, years=[2012, 2013, 2014, 2016, 2017, 2018, 2020],
            frames_per_day=frames_per_day, use_qd=use_aux_qd,
        )
        declared_geometry = config.get("geometry_provenance")
        if isinstance(declared_geometry, (str, os.PathLike)):
            with open(declared_geometry, encoding="utf-8") as stream:
                declared_geometry = json.load(stream)
        if declared_geometry is not None:
            for key in ("feature_version", "grid_sha256", "frames_per_day"):
                if declared_geometry.get(key) != geometry_provenance[key]:
                    raise ValueError(f"Declared geometry provenance disagrees with selected assets: {key}")
            for year, meta in geometry_provenance["cache_by_year"].items():
                if declared_geometry.get("cache_by_year", {}).get(year) != meta:
                    raise ValueError(f"Declared geometry cache differs in {year}")
    train_stride_cfg = config.get("train_sample_stride")
    val_stride_cfg = config.get("val_sample_stride")
    train_sample_stride = (
        int(train_stride_cfg)
        if isinstance(train_stride_cfg, int) and train_stride_cfg > 0
        else predict_step
    )
    val_sample_stride = (
        int(val_stride_cfg)
        if isinstance(val_stride_cfg, int) and val_stride_cfg > 0
        else predict_step
    )
    loader_workers = int(config.get("loader_workers", 4))
    loader_persistent = bool(config.get("loader_persistent", True))
    train_max_samples_cfg = config.get("train_max_samples")
    val_max_samples_cfg = config.get("val_max_samples")
    train_max_samples = (
        int(train_max_samples_cfg)
        if isinstance(train_max_samples_cfg, int) and train_max_samples_cfg > 0
        else None
    )
    val_max_samples = (
        int(val_max_samples_cfg)
        if isinstance(val_max_samples_cfg, int) and val_max_samples_cfg > 0
        else None
    )

    # 气候态残差开关
    use_climatology = bool(config.get("use_climatology", False))
    climatology_path = config.get("climatology_path")
    if use_climatology and not climatology_path:
        climatology_path = os.path.join(memmap_root, "climatology.npy")
    print(f"[气候态残差学习] {'启用' if use_climatology else '未启用'}")

    # 数据集实例化
    train_dataset = TEC_Dataset(
        memmap_root, time_step=time_step, predict_step=predict_step,
        train=True,
        sample_stride=train_sample_stride, max_samples=train_max_samples,
        climatology_path=climatology_path if use_climatology else None,
        frames_per_day=frames_per_day, use_aux_sza=use_aux_sza, use_aux_qd=use_aux_qd,
    )
    val_dataset = TEC_Dataset(
        memmap_root, time_step=time_step, predict_step=predict_step,
        validation=True,
        sample_stride=val_sample_stride, max_samples=val_max_samples,
        climatology_path=climatology_path if use_climatology else None,
        frames_per_day=frames_per_day, use_aux_sza=use_aux_sza, use_aux_qd=use_aux_qd,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        pin_memory=True, num_workers=loader_workers,
        persistent_workers=loader_persistent and loader_workers > 0,
        worker_init_fn=_seed_worker,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=loader_workers,
        persistent_workers=loader_persistent and loader_workers > 0,
        worker_init_fn=_seed_worker,
    )

    # SimVP模型
    model_suffix = str(config.get("model_suffix", "stage_i"))
    hidden_dim = int(config.get("hidden_dim", 32))
    n_layers = int(config.get("n_layers", 2))
    attn_name = str(config.get("attn_name", "none"))
    model = SimVP(
        input_dim=1 + int(use_aux_sza) + 3 * int(use_aux_qd),
        hidden_dim=hidden_dim,
        output_dim=1,
        time_step=time_step,
        predict_step=predict_step,
        n_layers=n_layers,
        attn_name=attn_name,
    ).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 优化器与学习率
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler_plateau = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=lr * 0.0001
    )
    criterion = nn.SmoothL1Loss()

    # 归一化参数
    # Correct inverse transform must match actual loader normalization.
    tec_min = float(train_dataset.global_min)
    tec_max = float(train_dataset.global_max)
    if not np.isclose(val_dataset.global_max, tec_max):
        raise ValueError("Train/validation normalization mismatch")
    tec_scale = max(tec_max - tec_min, 1e-8)

    os.makedirs(model_save_dir, exist_ok=True)
    if os.path.exists(os.path.join(model_save_dir, f"best_SimVP_{model_suffix}.pth")):
        raise FileExistsError("Existing corrected checkpoint; choose a new suffix/directory")
    model_save_path = os.path.join(model_save_dir, f"best_SimVP_{model_suffix}.pth")
    best_val_rmse = float('inf')
    best_epoch = 0
    early_stop_count = 0
    train_loss_history = []
    val_rmse_history = []
    val_rmse_global_history = []

    print("开始训练 SimVP ...")
    print("=" * 80)
    for epoch in range(epochs):
        current_epoch = epoch + 1
        model.train()
        train_loss = 0.0
        loop = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f"Epoch {current_epoch}/{epochs}")

        for batch_idx, batch in loop:
            # 按开关解包batch
            if use_climatology:
                inputs, targets, _ = batch
            else:
                inputs, targets = batch
            
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                outputs = model(inputs)
                # 残差模式输出可正可负，不裁剪；直接模式按开关裁剪
                outputs_for_loss = torch.clamp(outputs, min=0) if (enforce_nonneg and not use_climatology) else outputs
                total_loss = criterion(outputs_for_loss, targets)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += total_loss.item() * inputs.size(0)
            loop.set_postfix(loss=total_loss.item())

        train_loss /= len(train_loader.dataset)
        train_loss_history.append(float(train_loss))

        # 验证
        model.eval()
        val_sse = 0.0
        val_count = 0
        val_rmse_batch_sum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if use_climatology:
                    inputs, targets, targets_clim = batch
                    targets_clim = targets_clim.to(device)
                else:
                    inputs, targets = batch

                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)

                # 残差模式还原回原始TEC再算指标
                if use_climatology:
                    outputs_full = outputs + targets_clim
                    targets_full = targets + targets_clim
                else:
                    outputs_full = outputs
                    targets_full = targets

                if enforce_nonneg:
                    outputs_full = torch.clamp(outputs_full, min=0)

                outputs_denorm = outputs_full * tec_scale + tec_min
                targets_denorm = targets_full * tec_scale + tec_min
                rmse_batch = torch.sqrt(nn.MSELoss()(outputs_denorm, targets_denorm))
                val_rmse_batch_sum += rmse_batch.item() * inputs.size(0)
                diff = (outputs_denorm - targets_denorm).float()
                val_sse += float(torch.sum(diff * diff).item())
                val_count += int(diff.numel())

        val_rmse_batch = val_rmse_batch_sum / len(val_loader.dataset)
        val_rmse_global = float(np.sqrt(val_sse / max(1, val_count)))
        metric_to_opt = val_rmse_batch if val_metric == "batch" else val_rmse_global
        scheduler_plateau.step(metric_to_opt)
        val_rmse_history.append(float(val_rmse_batch))
        val_rmse_global_history.append(val_rmse_global)

        print(
            f"Epoch {current_epoch}: Train Loss={train_loss:.4f}, "
            f"Val RMSE[batch]={val_rmse_batch:.4f}, "
            f"Val RMSE[global]={val_rmse_global:.4f}, "
            f"Metric[{val_metric}]={metric_to_opt:.4f}"
        )

        if metric_to_opt < best_val_rmse:
            best_val_rmse = float(metric_to_opt)
            best_epoch = current_epoch
            torch.save(model.state_dict(), model_save_path)
            early_stop_count = 0
            print(f"  >> Best model saved! RMSE={best_val_rmse:.4f}")
        else:
            early_stop_count += 1
            if early_stop_count >= patience:
                print("Early stopping!")
                break

    print("=" * 80)
    print(f"训练完成! Best Epoch={best_epoch}, Best Val RMSE={best_val_rmse:.4f}")

    train_metrics = {
        'best_epoch': best_epoch,
        'best_val_rmse': best_val_rmse,
        'train_loss': train_loss_history,
        'val_rmse': val_rmse_history,
        'val_rmse_history_definition': 'sample-weighted mean of batch RMSE (legacy field)',
        'val_rmse_global': val_rmse_global_history,
        'geometry': geometry_provenance,
        'normalization': {'min': tec_min, 'max': tec_max},
        'config': {
            'batch_size': batch_size, 'lr': lr, 'epochs': epochs,
            'patience': patience, 'val_metric': val_metric,
            'time_step': time_step, 'predict_step': predict_step,
            'frames_per_day': frames_per_day, 'use_aux_sza': use_aux_sza,
            'use_aux_qd': use_aux_qd,
            'hidden_dim': hidden_dim, 'n_layers': n_layers,
            'attn_name': attn_name,
            'use_climatology': use_climatology,
            'climatology_path': climatology_path,
            'train_sample_stride': train_sample_stride, 'val_sample_stride': val_sample_stride,
            'enforce_nonneg': enforce_nonneg, 'loader_workers': loader_workers,
            'model_suffix': model_suffix,
        },
    }
    metrics_path = os.path.join(model_save_dir, f"train_metrics_SimVP_{model_suffix}.json")
    with open(metrics_path, 'w') as f:
        json.dump(train_metrics, f, indent=2)
    return best_val_rmse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--model-save-dir", default="models")
    parser.add_argument("--grid-npz", help="Default: DATA_ROOT/grid.npz")
    parser.add_argument("--geometry-cache", help="Default: DATA_ROOT/geometry")
    parser.add_argument("--geometry-provenance", help="Optional JSON to cross-check against selected geometry assets")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--time-step", type=int, default=24)
    parser.add_argument("--predict-step", type=int, default=24)
    parser.add_argument("--frames-per-day", type=int, default=96)
    parser.add_argument("--use-aux-sza", type=int, default=0)
    parser.add_argument("--use-aux-qd", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--train-sample-stride", type=int, default=-1)
    parser.add_argument("--val-sample-stride", type=int, default=-1)
    parser.add_argument("--loader-workers", type=int, default=4)
    parser.add_argument("--loader-persistent", type=int, default=1)
    parser.add_argument("--train-max-samples", type=int, default=-1)
    parser.add_argument("--val-max-samples", type=int, default=-1)
    parser.add_argument("--cache-dir", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--val-metric", type=str, default="global", choices=["batch", "global"])
    parser.add_argument("--enforce-nonneg", type=int, default=0)
    parser.add_argument("--model-suffix", type=str, default="stage_i")
    parser.add_argument("--attn-name", type=str, default="none", choices=["none", "cbam", "simam"])
    # 气候态相关参数
    parser.add_argument("--use-climatology", type=int, default=0)
    parser.add_argument("--climatology-path", type=str, default=None)
    args = parser.parse_args()

    cfg = {
        "data_root": args.data_root,
        "model_save_dir": args.model_save_dir,
        "grid_npz": args.grid_npz,
        "geometry_cache": args.geometry_cache,
        "geometry_provenance": args.geometry_provenance,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "patience": args.patience,
        "time_step": args.time_step,
        "predict_step": args.predict_step,
        "frames_per_day": args.frames_per_day,
        "use_aux_sza": bool(args.use_aux_sza),
        "use_aux_qd": bool(args.use_aux_qd),
        "hidden_dim": args.hidden_dim,
        "n_layers": args.n_layers,
        "train_sample_stride": args.train_sample_stride,
        "val_sample_stride": args.val_sample_stride,
        "loader_workers": args.loader_workers,
        "loader_persistent": bool(args.loader_persistent),
        "train_max_samples": args.train_max_samples,
        "val_max_samples": args.val_max_samples,
        "val_metric": args.val_metric,
        "enforce_nonneg": bool(args.enforce_nonneg),
        "model_suffix": args.model_suffix,
        "attn_name": args.attn_name,
        "use_climatology": bool(args.use_climatology),
        "climatology_path": args.climatology_path,
    }
    train_simvp_model(cfg)
