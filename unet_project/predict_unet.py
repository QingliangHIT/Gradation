import os
import cv2
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from unet_model import UNet, ResUNet, NNUNet, NNUNetv2


def load_model(model_path, model_name, device):
    models = {"unet": UNet, "resunet": ResUNet, "nnunet": NNUNet, "nnunetv2": NNUNetv2}
    model = models[model_name](in_channels=3, num_classes=2).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    state = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded model: {model_name} from {model_path}")
    return model


def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, flags)
    return img


def preprocess_image(img_path, img_size):
    img = imread_unicode(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    img = cv2.resize(img, (img_size, img_size))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return img_rgb, img_tensor


def predict(model, img_tensor, device):
    model.eval()
    with torch.no_grad():
        output = model(img_tensor.to(device))
        return torch.argmax(output, dim=1).squeeze(0).cpu().numpy()


def make_overlay(img_rgb, pred):
    overlay = img_rgb.astype(np.float32) / 255.0
    overlay[:, :, 0] = np.where(pred == 1, 1.0, overlay[:, :, 0])
    overlay[:, :, 1] = np.where(pred == 1, 0.0, overlay[:, :, 1])
    overlay[:, :, 2] = np.where(pred == 1, 0.0, overlay[:, :, 2])
    return np.clip(overlay * 0.6 + img_rgb.astype(np.float32) / 255.0 * 0.4, 0, 1)


def compute_metrics(pred, mask):
    if mask is None:
        return None, None, None
    mask = (mask > 127).astype(np.int64)
    tp = ((pred == 1) & (mask == 1)).sum()
    fp = ((pred == 1) & (mask == 0)).sum()
    fn = ((pred == 0) & (mask == 1)).sum()
    tn = ((pred == 0) & (mask == 0)).sum()
    iou = tp / (tp + fp + fn + 1e-8)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    acc = (tp + tn) / (tp + fp + fn + tn + 1e-8)
    return iou, dice, acc


def select_folder_dialog(title="Select Folder"):
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title=title)
    root.destroy()
    return folder if folder else None


def select_file_dialog(title="Select Image"):
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    filepath = filedialog.askopenfilename(
        title=title, filetypes=[("Image files", "*.png *.jpg *.bmp *.jpeg"), ("All files", "*.*")]
    )
    root.destroy()
    return filepath if filepath else None


IMAGE_EXTS = (".png", ".jpg", ".bmp", ".jpeg")


class PredictorApp:
    ZOOM_MIN = 0.5
    ZOOM_MAX = 8.0
    ZOOM_STEP = 0.25
    IMG_AXES_NAMES = ("ax_orig", "ax_pred", "ax_overlay")

    def __init__(self, model, device, img_dir, mask_dir, img_size):
        self.model = model
        self.device = device
        self.img_size = img_size
        self.img_dir = img_dir
        self.mask_dir = mask_dir

        self.img_files = []
        self.mask_files = []
        self.current_idx = 0
        self.total = 0
        self.single_mode = False
        self.zoom = 1.0

        self._img_rgb = None
        self._pred = None
        self._overlay = None
        self._mask = None

        self._dragging = False
        self._drag_pixel = None   # (px, py) 像素坐标
        self._drag_xlim = None
        self._drag_ylim = None
        self._drag_ax = None      # 拖拽起始的 axes

        self._setup_figure()
        self._setup_buttons()
        self._connect_events()

        if img_dir and os.path.exists(img_dir):
            self._load_directory(img_dir, mask_dir if mask_dir and os.path.exists(mask_dir) else None)
        else:
            self._show_welcome()

    def _setup_figure(self):
        self.fig = plt.figure(figsize=(16, 8))
        self.fig.canvas.manager.set_window_title("UNet/ResUNet Predictor")

        gs = self.fig.add_gridspec(3, 4, hspace=0.35, wspace=0.15, height_ratios=[1, 0.06, 0.06])
        self.ax_orig = self.fig.add_subplot(gs[0, 0])
        self.ax_pred = self.fig.add_subplot(gs[0, 1])
        self.ax_overlay = self.fig.add_subplot(gs[0, 2])
        self.ax_info = self.fig.add_subplot(gs[0, 3])
        self.ax_zoom = self.fig.add_subplot(gs[1, :])

        self.img_axes = [self.ax_orig, self.ax_pred, self.ax_overlay]
        self.all_axes = self.img_axes + [self.ax_info]

        for ax in self.all_axes:
            ax.axis("off")

        self.ax_orig.set_title("Original", fontsize=12)
        self.ax_pred.set_title("Prediction", fontsize=12)
        self.ax_overlay.set_title("Overlay", fontsize=12)
        self.ax_info.set_title("Info", fontsize=12)

        self.zoom_slider = Slider(self.ax_zoom, "Zoom", self.ZOOM_MIN, self.ZOOM_MAX,
                                  valinit=1.0, valstep=0.25)
        self.zoom_slider.on_changed(self._on_zoom_change)

    def _setup_buttons(self):
        bw, bh = 0.065, 0.035
        gap = 0.01
        # Row 1: Dir, File, MDir, MFile, Save - 居中
        row1_y = 0.055
        row1_btns = [
            ("Dir", "_on_open_dir"),
            ("File", "_on_open_img"),
            ("MDir", "_on_mask_dir"),
            ("MFile", "_on_mask_file"),
            ("Save", "_on_save"),
            ("◀ Prev", "_on_prev"), ("Next ▶", "_on_next"),
        ]
        total_w = len(row1_btns) * bw + (len(row1_btns) - 1) * gap
        start_x = (1.0 - total_w) / 2
        self._buttons = []
        for i, (label, handler) in enumerate(row1_btns):
            x = start_x + i * (bw + gap)
            ax = self.fig.add_axes([x, row1_y, bw, bh])
            btn = Button(ax, label)
            btn.on_clicked(getattr(self, handler))
            self._buttons.append(btn)

    def _connect_events(self):
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.fig.canvas.mpl_connect("button_press_event", self._on_button_press)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)

    # ── Welcome ──────────────────────────────────────────────

    def _show_welcome(self):
        for ax in self.all_axes:
            ax.clear()
            ax.axis("off")
        for ax, title in zip(self.img_axes, ["Original", "Prediction", "Overlay"]):
            ax.set_title(title, fontsize=12)
        self.ax_info.set_title("Info", fontsize=12)

        text = (
            "=== Welcome ===\n\n"
            "--- Load ---\n"
            "  [Dir]    Open image folder\n"
            "  [File]   Open single image\n"
            "  [MDir]   Set mask folder\n"
            "  [MFile]  Set single mask\n\n"
            "--- View ---\n"
            "  Slider / Scroll: Zoom\n"
            "  Drag: Pan (when zoomed)\n"
            "  DblClick: Reset view\n"
            "  ← → : Navigate\n"
            "  S : Save result"
        )
        self.ax_info.text(0.05, 0.95, text, transform=self.ax_info.transAxes,
                          fontsize=9, va="top", family="monospace")
        self.fig.canvas.draw_idle()

    # ── Loading ──────────────────────────────────────────────

    def _load_directory(self, img_dir, mask_dir=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.single_mode = False
        self.img_files = sorted(
            os.path.join(img_dir, f) for f in os.listdir(img_dir)
            if f.lower().endswith(IMAGE_EXTS)
        )
        self.mask_files = []
        if mask_dir and os.path.exists(mask_dir):
            self.mask_files = sorted(
                os.path.join(mask_dir, f) for f in os.listdir(mask_dir)
                if f.lower().endswith(IMAGE_EXTS)
            )
        self.total = len(self.img_files)
        self.current_idx = 0
        if self.total == 0:
            print(f"No images found in {img_dir}")
            return
        tag = "with mask" if self.mask_files else "no mask"
        print(f"Loaded {self.total} images from {img_dir} ({tag})")
        self._update_display()

    def _load_single_file(self, img_path, mask_path=None):
        self.single_mode = True
        self.img_files = [img_path]
        self.mask_files = [mask_path] if mask_path else []
        self.total = 1
        self.current_idx = 0
        tag = "with mask" if mask_path else "no mask"
        print(f"Loaded single image: {img_path} ({tag})")
        self._update_display()

    # ── Button handlers ──────────────────────────────────────

    def _on_open_dir(self, event):
        img_dir = select_folder_dialog("Select Image Directory")
        if img_dir:
            self._load_directory(img_dir)

    def _on_open_img(self, event):
        img_path = select_file_dialog("Select Image File")
        if img_path:
            self._load_single_file(img_path)

    def _on_mask_dir(self, event):
        if self.total == 0:
            return
        path = select_folder_dialog("Select Mask Directory")
        if path:
            self.mask_dir = path
            self.mask_files = sorted(
                os.path.join(path, f) for f in os.listdir(path)
                if f.lower().endswith(IMAGE_EXTS)
            )
            self._update_display()

    def _on_mask_file(self, event):
        if self.total == 0:
            return
        mask_path = select_file_dialog("Select Mask File")
        if mask_path:
            mask = imread_unicode(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                self._mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
                if self.single_mode:
                    self.mask_files = [mask_path]
                self._update_display()

    def _on_prev(self, event):
        if self.total > 0:
            self.current_idx = (self.current_idx - 1) % self.total
            self._update_display()

    def _on_next(self, event):
        if self.total > 0:
            self.current_idx = (self.current_idx + 1) % self.total
            self._update_display()

    def _on_save(self, event):
        if self.total == 0:
            return
        save_dir = "predictions"
        os.makedirs(save_dir, exist_ok=True)
        fname = os.path.basename(self.img_files[self.current_idx])
        name, _ = os.path.splitext(fname)

        self.fig.savefig(os.path.join(save_dir, f"{name}_preview.png"), bbox_inches="tight", dpi=150)
        cv2.imwrite(os.path.join(save_dir, f"{name}_pred.png"), (self._pred * 255).astype(np.uint8))
        overlay_bgr = cv2.cvtColor((self._overlay * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_dir, f"{name}_overlay.png"), overlay_bgr)
        print(f"Saved: {save_dir}/{name}_preview.png, {name}_pred.png, {name}_overlay.png")

    # ── Zoom & Pan ───────────────────────────────────────────

    def _set_zoom(self, val):
        self.zoom = val
        self.zoom_slider.set_val(val)

    def _on_zoom_change(self, val):
        self.zoom = val
        self._apply_zoom_to_axes()

    def _on_scroll(self, event):
        if self.total == 0:
            return
        if event.button == "up":
            self._set_zoom(min(self.zoom + self.ZOOM_STEP, self.ZOOM_MAX))
        elif event.button == "down":
            self._set_zoom(max(self.zoom - self.ZOOM_STEP, self.ZOOM_MIN))

    def _on_button_press(self, event):
        if event.inaxes not in self.img_axes:
            return
        if event.dblclick:
            self._set_zoom(1.0)
            return
        if event.button == 1 and self.zoom > 1.0:
            self._dragging = True
            self._drag_pixel = (event.x, event.y)
            self._drag_ax = event.inaxes
            self._drag_xlim = event.inaxes.get_xlim()
            self._drag_ylim = event.inaxes.get_ylim()

    def _on_release(self, event):
        self._dragging = False
        self._drag_pixel = None
        self._drag_ax = None

    def _on_motion(self, event):
        if not self._dragging or self._drag_pixel is None or self._drag_ax is None:
            return
        if event.x is None or event.y is None:
            return
        # 像素偏移
        dpx = event.x - self._drag_pixel[0]
        dpy = event.y - self._drag_pixel[1]
        # 将像素偏移转换为数据坐标偏移
        ax = self._drag_ax
        xlim = self._drag_xlim
        ylim = self._drag_ylim
        bbox = ax.get_window_extent()
        data_w = xlim[1] - xlim[0]
        data_h = ylim[0] - ylim[1]  # ylim 是反的（上小下大）
        dx = dpx * data_w / bbox.width
        dy = -dpy * data_h / bbox.height  # 屏幕 y 向下，数据 y 向上
        for a in self.img_axes:
            a.set_xlim(xlim[0] - dx, xlim[1] - dx)
            a.set_ylim(ylim[0] - dy, ylim[1] - dy)
        self.fig.canvas.draw_idle()

    def _apply_zoom_to_axes(self):
        if self._img_rgb is None:
            return
        h, w = self._img_rgb.shape[:2]
        cx, cy = w / 2, h / 2
        half_w, half_h = w / (2 * self.zoom), h / (2 * self.zoom)
        for ax in self.img_axes:
            ax.set_xlim(cx - half_w, cx + half_w)
            ax.set_ylim(cy + half_h, cy - half_h)
            ax.set_title(f"{ax.get_title().split(' (')[0]} ({self.zoom:.1f}x)", fontsize=12)
        self._update_info_text()
        self.fig.canvas.draw_idle()

    # ── Keyboard ─────────────────────────────────────────────

    def _on_key(self, event):
        actions = {
            "left": self._on_prev,
            "right": self._on_next,
            "s": self._on_save,
            "d": self._on_open_dir,
            "f": self._on_open_img,
        }
        if event.key in actions:
            actions[event.key](None)
        elif event.key == "+":
            self._set_zoom(min(self.zoom + self.ZOOM_STEP, self.ZOOM_MAX))
        elif event.key == "-":
            self._set_zoom(max(self.zoom - self.ZOOM_STEP, self.ZOOM_MIN))

    # ── Display ──────────────────────────────────────────────

    def _get_mask_path(self):
        if self.total == 0:
            return None
        fname = os.path.basename(self.img_files[self.current_idx])
        if self.single_mode:
            return self.mask_files[0] if self.mask_files else None
        if self.mask_dir and os.path.exists(self.mask_dir):
            candidate = os.path.join(self.mask_dir, fname)
            if os.path.exists(candidate):
                return candidate
            if self.mask_files and self.current_idx < len(self.mask_files):
                return self.mask_files[self.current_idx]
        return None

    def _update_info_text(self):
        if self.total == 0:
            return
        fname = os.path.basename(self.img_files[self.current_idx])
        mode = "[Single File]" if self.single_mode else f"[Dir: {os.path.basename(self.img_dir)}]"
        text = f"{mode}\nFile: {fname}\nIndex: {self.current_idx + 1}/{self.total}\nZoom: {self.zoom:.2f}x\n"
        if self._mask is not None:
            iou, dice, acc = compute_metrics(self._pred, self._mask)
            text += f"\n--- Metrics ---\nIoU:  {iou:.4f}\nDice: {dice:.4f}\nAcc:  {acc:.4f}"
        else:
            text += "\n(No GT mask)"
        text += "\n\n--- Keys ---\nD   : Open directory\nF   : Open image\n+/- : Zoom\n← → : Navigate\nS   : Save"
        if self.ax_info.texts:
            self.ax_info.texts[0].set_text(text)
        else:
            self.ax_info.text(0.05, 0.95, text, transform=self.ax_info.transAxes,
                              fontsize=9, va="top", family="monospace")

    def _update_display(self):
        if self.total == 0:
            return

        img_path = self.img_files[self.current_idx]
        img_rgb, img_tensor = preprocess_image(img_path, self.img_size)

        mask_path = self._get_mask_path()
        self._mask = None
        if mask_path and os.path.exists(mask_path):
            mask = imread_unicode(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                self._mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        self._img_rgb = img_rgb
        self._pred = predict(self.model, img_tensor, self.device)
        self._overlay = make_overlay(img_rgb, self._pred)

        for ax in self.all_axes:
            ax.clear()
            ax.axis("off")

        self.ax_orig.set_title(f"Original ({self.zoom:.1f}x)", fontsize=12)
        self.ax_pred.set_title(f"Prediction ({self.zoom:.1f}x)", fontsize=12)
        self.ax_overlay.set_title(f"Overlay ({self.zoom:.1f}x)", fontsize=12)
        self.ax_info.set_title("Info", fontsize=12)

        self.ax_orig.imshow(self._img_rgb)
        self.ax_pred.imshow(self._pred, cmap="gray")
        self.ax_overlay.imshow(self._overlay)

        self._apply_zoom_to_axes()
        self._update_info_text()
        self.fig.canvas.draw_idle()

    def run(self):
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive model prediction viewer")
    parser.add_argument("--model-path", type=str, default=r"A:\05-Codes\Gradation\DATA\resunet_best.pth",
                        help="Path to trained model (.pth)")
    parser.add_argument("--model", type=str, default="resunet", choices=["unet", "resunet", "nnunet", "nnunetv2"],
                        help="Model architecture")
    parser.add_argument("--img-dir", type=str, default="particle/val/images",
                        help="Directory containing images to predict")
    parser.add_argument("--mask-dir", type=str, default="particle/val/masks",
                        help="Directory containing GT masks (optional, for metrics)")
    parser.add_argument("--img-size", type=int, default=256,
                        help="Image size for model input")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(args.model_path, args.model, device)

    app = PredictorApp(
        model=model,
        device=device,
        img_dir=args.img_dir,
        mask_dir=args.mask_dir,
        img_size=args.img_size,
    )
    app.run()
