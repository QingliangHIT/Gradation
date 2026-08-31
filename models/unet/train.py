# -*- coding: utf-8 -*-
"""UNet 系列模型训练脚本。

用法:
    python -m models.unet.train --model resunet --data-dir particle --epochs 100
"""
import csv
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.unet.dataset import ParticleDataset
from models.unet.metrics import compute_metrics_tensor
from models.unet import NNUNet, NNUNetv2, ResUNet, UNet


def save_val_samples(model, val_loader, device, save_dir, epoch, num_samples=4):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    count = 0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            imgs_np = imgs.cpu().numpy()
            masks_np = masks.numpy()

            for i in range(imgs.size(0)):
                if count >= num_samples:
                    return
                idx = count + 1

                img = imgs_np[i].transpose(1, 2, 0)
                img = np.clip(img, 0, 1)
                if img.shape[2] == 1:
                    img = np.squeeze(img, axis=2)
                    img_show = np.stack([img] * 3, axis=-1)
                else:
                    img_show = img.copy()

                gt = masks_np[i]
                pred = preds[i]

                overlay = img_show.copy()
                overlay[:, :, 0] = np.where(pred == 1, 1.0, overlay[:, :, 0])
                overlay[:, :, 1] = np.where(pred == 1, 0.0, overlay[:, :, 1])
                overlay[:, :, 2] = np.where(pred == 1, 0.0, overlay[:, :, 2])
                overlay = np.clip(overlay * 0.6 + img_show * 0.4, 0, 1)

                fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                axes[0].imshow(img_show, cmap="gray" if len(img_show.shape) == 2 or img_show.shape[2] == 1 else None)
                axes[0].set_title("Original")
                axes[0].axis("off")

                axes[1].imshow(gt, cmap="gray")
                axes[1].set_title("Ground Truth")
                axes[1].axis("off")

                axes[2].imshow(pred, cmap="gray")
                axes[2].set_title("Prediction")
                axes[2].axis("off")

                axes[3].imshow(overlay)
                axes[3].set_title("Overlay (Red=Pred)")
                axes[3].axis("off")

                fig.suptitle(f"Epoch {epoch} - Sample {idx}", fontsize=14)
                fig.tight_layout()
                fig.savefig(os.path.join(save_dir, f"val_epoch{epoch:03d}_sample{idx}.png"), dpi=120)
                plt.close(fig)
                count += 1


def plot_training_curves(history, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(epochs, history["train_loss"], "b-o", markersize=3, label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r-o", markersize=3, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, history["val_iou"], "g-o", markersize=3, label="IoU")
    ax.plot(epochs, history["val_dice"], "b-s", markersize=3, label="Dice")
    ax.plot(epochs, history["val_acc"], "r-^", markersize=3, label="Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("Validation Metrics")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "metrics_curve.png"), dpi=150)
    plt.close(fig)

    if "lr" in history and len(history["lr"]) > 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(epochs, history["lr"], "m-o", markersize=3)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "lr_curve.png"), dpi=150)
        plt.close(fig)


def get_next_output_dir(model_name):
    base_dir = os.path.join("runs", model_name)
    if not os.path.exists(base_dir):
        return base_dir
    idx = 2
    while True:
        candidate = os.path.join("runs", f"{model_name}-{idx}")
        if not os.path.exists(candidate):
            return candidate
        idx += 1


def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_iou, val_dice, val_acc = 0.0, 0.0, 0.0
    val_count = 0
    val_pbar = tqdm(val_loader, desc="[Pre-Validate]", leave=False)

    with torch.no_grad():
        for imgs, masks in val_pbar:
            imgs = imgs.to(device)
            masks = masks.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, masks)

            bs = imgs.size(0)
            val_loss += loss.item() * bs
            iou, dice, acc = compute_metrics_tensor(outputs, masks)
            val_iou += iou * bs
            val_dice += dice * bs
            val_acc += acc * bs
            val_count += bs

            val_pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg": f"{val_loss / val_count:.4f}",
                "iou": f"{val_iou / val_count:.4f}",
                "dice": f"{val_dice / val_count:.4f}",
                "acc": f"{val_acc / val_count:.4f}",
            })

    val_loss /= len(val_loader.dataset)
    val_iou /= len(val_loader.dataset)
    val_dice /= len(val_loader.dataset)
    val_acc /= len(val_loader.dataset)

    return val_loss, val_iou, val_dice, val_acc


def train(
    data_dir,
    model_name,
    pretrained,
    epochs,
    batch_size,
    img_size,
    lr,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = get_next_output_dir(model_name)
    os.makedirs(output_dir, exist_ok=True)
    val_sample_dir = os.path.join(output_dir, "val_samples")
    os.makedirs(val_sample_dir, exist_ok=True)

    best_model_path = os.path.join(output_dir, "best_model.pth")

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    train_dataset = ParticleDataset(train_dir, img_size)
    val_dataset = ParticleDataset(val_dir, img_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    models = {"unet": UNet, "resunet": ResUNet, "nnunet": NNUNet, "nnunetv2": NNUNetv2}
    model = models[model_name](in_channels=3, num_classes=2).to(device)
    print(f"Using model: {model_name}")
    print(f"Output dir: {output_dir}")

    criterion = nn.CrossEntropyLoss()
    # nnUNet / nnUNetv2 深度监督权重（指数衰减）
    ds_weights = None
    if model_name == "nnunetv2":
        num_ds = model.num_stages  # 解码器级数
        ds_weights = [0.5 ** i for i in range(num_ds)]
        ds_weights = [w / sum(ds_weights) for w in ds_weights]  # 归一化
        print(f"Deep supervision weights: {[f'{w:.4f}' for w in ds_weights]}")
    elif model_name == "nnunet":
        # NNUNet 深度监督输出: [out, ds2, ds3, ds4, ds5] 共5个
        num_ds = 5
        ds_weights = [0.5 ** i for i in range(num_ds)]
        ds_weights = [w / sum(ds_weights) for w in ds_weights]
        print(f"Deep supervision weights: {[f'{w:.4f}' for w in ds_weights]}")

    best_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_iou": [], "val_dice": [], "val_acc": [], "lr": []}

    if pretrained and os.path.exists(pretrained):
        checkpoint = torch.load(pretrained, map_location=device, weights_only=True)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded pretrained model from: {pretrained}")

        print("Validating pretrained model...")
        pre_loss, pre_iou, pre_dice, pre_acc = validate(model, val_loader, criterion, device)
        print(f"Pretrained baseline => Val Loss: {pre_loss:.4f} | IoU: {pre_iou:.4f} | Dice: {pre_dice:.4f} | Acc: {pre_acc:.4f}")
        best_loss = pre_loss
    elif pretrained:
        print(f"[WARNING] pretrained model not found: {pretrained}, training from scratch")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_count = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]", leave=False)

        for imgs, masks in train_pbar:
            imgs = imgs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)

            # 深度监督损失
            if isinstance(outputs, list) and ds_weights is not None:
                loss = sum(w * criterion(o, masks) for w, o in zip(ds_weights, outputs))
            else:
                loss = criterion(outputs, masks)

            loss.backward()
            optimizer.step()

            bs = imgs.size(0)
            train_loss += loss.item() * bs
            train_count += bs
            avg_loss = train_loss / train_count

            postfix = {"loss": f"{loss.item():.4f}", "avg": f"{avg_loss:.4f}"}
            if torch.cuda.is_available():
                postfix["gpu"] = f"{torch.cuda.memory_allocated() / 1024 ** 2:.0f}MB"
            train_pbar.set_postfix(postfix)

        train_loss /= len(train_dataset)
        history["train_loss"].append(train_loss)

        val_loss, val_iou, val_dice, val_acc = validate(model, val_loader, criterion, device)

        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou)
        history["val_dice"].append(val_dice)
        history["val_acc"].append(val_acc)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        history["lr"].append(current_lr)

        print(f"Epoch {epoch + 1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"IoU: {val_iou:.4f} | "
              f"Dice: {val_dice:.4f} | "
              f"Acc: {val_acc:.4f} | "
              f"LR: {current_lr:.2e}")

        if (epoch + 1) % 5 == 0 or epoch == 0:
            save_val_samples(model, val_loader, device, val_sample_dir, epoch + 1, num_samples=4)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> Saved best model (val_loss={best_loss:.4f}, IoU={val_iou:.4f}, Dice={val_dice:.4f})")

        plot_training_curves(history, output_dir)
        with open(os.path.join(output_dir, "history.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "val_iou", "val_dice", "val_acc", "lr"])
            for i in range(len(history["train_loss"])):
                writer.writerow([
                    i + 1,
                    f"{history['train_loss'][i]:.6f}",
                    f"{history['val_loss'][i]:.6f}",
                    f"{history['val_iou'][i]:.6f}",
                    f"{history['val_dice'][i]:.6f}",
                    f"{history['val_acc'][i]:.6f}",
                    f"{history['lr'][i]:.6e}",
                ])

    print("Training done.")
    print(f"Best Val Loss: {best_loss:.4f}")
    print(
        f"Final Val IoU: {history['val_iou'][-1]:.4f}, Dice: {history['val_dice'][-1]:.4f}, Acc: {history['val_acc'][-1]:.4f}")
    print(f"Results saved to: {output_dir}/")

    torch.save({"model_state_dict": model.state_dict(), "history": history, "model_name": model_name},
               os.path.join(output_dir, "last_checkpoint.pth"))

    return model, history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resunet", choices=["unet", "resunet", "nnunet", "nnunetv2"],
                        help="Model architecture")
    parser.add_argument("--data-dir", type=str, default="particle",
                        help="Dataset root directory")
    parser.add_argument("--pretrained", type=str, default='resunet_best.pth',
                        help="Path to pretrained model (.pth). If set, validates it first and new training must beat it")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for training and validation")
    parser.add_argument("--img-size", type=int, default=256,
                        help="Input image size (will be resized to this)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Initial learning rate")
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        model_name=args.model,
        pretrained=args.pretrained,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        lr=args.lr,
    )
