"""
inference.py — Single image or batch inference
Usage:
  python src/inference.py --input satellite.jpg --output map.png --model_dir saved_models/best_generator
  python src/inference.py --input_dir data/test/ --output_dir outputs/predictions/ --model_dir saved_models/best_generator
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image


# ─────────────────────────────────────────────
# PRE / POST PROCESSING
# ─────────────────────────────────────────────

def preprocess(image_path: str) -> tf.Tensor:
    """Load a JPEG/PNG, resize to 256x256, normalize to [-1, 1]."""
    img = tf.io.read_file(image_path)
    # Handle both JPEG and PNG
    try:
        img = tf.image.decode_jpeg(img, channels=3)
    except Exception:
        img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [256, 256])
    img = tf.cast(img, tf.float32)
    return (img / 127.5) - 1.0                  # [-1, 1]


def preprocess_pil(pil_image: Image.Image) -> tf.Tensor:
    """Preprocess a PIL Image object (used by API server)."""
    img = pil_image.convert('RGB').resize((256, 256))
    arr = np.array(img, dtype=np.float32)
    return (arr / 127.5) - 1.0                  # [-1, 1]


def postprocess(tensor: tf.Tensor) -> Image.Image:
    """Convert model output tensor to PIL Image."""
    img = (tensor[0].numpy() + 1.0) * 127.5     # [0, 255]
    img = np.clip(img, 0, 255).astype(np.uint8)
    return Image.fromarray(img)


def postprocess_with_comparison(sat_tensor, gen_tensor, real_tensor=None) -> Image.Image:
    """Create a side-by-side comparison image."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    panels = [('Satellite Input', sat_tensor[0]),
              ('Generated Map',   gen_tensor[0])]
    if real_tensor is not None:
        panels.append(('Ground Truth', real_tensor[0]))

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5))
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, img) in zip(axes, panels):
        ax.imshow((img.numpy() * 0.5 + 0.5).clip(0, 1))
        ax.set_title(title, fontsize=12)
        ax.axis('off')
    plt.tight_layout()

    # Render to PIL
    from io import BytesIO
    buf = BytesIO()
    plt.savefig(buf, format='PNG', dpi=120, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return Image.open(buf).copy()


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

def compute_metrics(real_img: np.ndarray, pred_img: np.ndarray) -> dict:
    """Compute MAE, SSIM, PSNR between two uint8 HxWx3 arrays."""
    try:
        from skimage.metrics import structural_similarity as ssim_fn
        from skimage.metrics import peak_signal_noise_ratio as psnr_fn
        mae  = float(np.mean(np.abs(real_img.astype(float) - pred_img.astype(float))) / 255.0)
        ssim = float(ssim_fn(real_img, pred_img, channel_axis=2, data_range=255))
        psnr = float(psnr_fn(real_img, pred_img, data_range=255))
        return {'MAE': mae, 'SSIM': ssim, 'PSNR': psnr}
    except ImportError:
        mae = float(np.mean(np.abs(real_img.astype(float) - pred_img.astype(float))) / 255.0)
        return {'MAE': mae}


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────

def load_generator(model_dir: str):
    """Load generator from SavedModel or TF checkpoint directory."""
    model_dir = Path(model_dir)
    # Try SavedModel format first
    saved_model_path = model_dir / 'saved_model.pb'
    if saved_model_path.exists() or (model_dir / 'assets').exists():
        print(f"[MODEL] Loading SavedModel from {model_dir}")
        return tf.saved_model.load(str(model_dir))
    # Try keras format
    keras_path = model_dir / 'generator.keras'
    if keras_path.exists():
        print(f"[MODEL] Loading Keras model from {keras_path}")
        return tf.keras.models.load_model(str(keras_path))
    raise FileNotFoundError(
        f"No SavedModel or .keras file found in {model_dir}.\n"
        f"Train first with: python src/train.py --export"
    )


# ─────────────────────────────────────────────
# SINGLE IMAGE INFERENCE
# ─────────────────────────────────────────────

def run_single(generator, input_path: str, output_path: str, compare: bool = False):
    """Run inference on a single satellite image."""
    inp = preprocess(input_path)[tf.newaxis]        # add batch dim
    out = generator(inp, training=False)
    
    if compare:
        result = postprocess_with_comparison(inp, out)
    else:
        result = postprocess(out)
    
    result.save(output_path)
    print(f"[OUT] Saved to {output_path}")
    return result


# ─────────────────────────────────────────────
# BATCH INFERENCE
# ─────────────────────────────────────────────

def run_batch(generator, input_dir: str, output_dir: str, compare: bool = False):
    """Run inference on all images in a directory."""
    os.makedirs(output_dir, exist_ok=True)
    input_paths = sorted(Path(input_dir).glob("*.jpg")) + \
                  sorted(Path(input_dir).glob("*.png"))
    
    if not input_paths:
        print(f"[WARN] No .jpg or .png images found in {input_dir}")
        return

    print(f"[BATCH] Processing {len(input_paths)} images...")
    all_metrics = []

    for img_path in input_paths:
        inp = preprocess(str(img_path))[tf.newaxis]
        out = generator(inp, training=False)

        stem = img_path.stem
        out_path = os.path.join(output_dir, f"{stem}_pred.png")

        if compare:
            result = postprocess_with_comparison(inp, out)
        else:
            result = postprocess(out)
        result.save(out_path)

        # If input is a paired image, compute metrics vs ground truth
        real_arr = np.array(Image.open(str(img_path)).resize((256, 256)).convert('RGB'))
        pred_arr = np.array(postprocess(out))
        metrics = compute_metrics(real_arr, pred_arr)
        all_metrics.append(metrics)
        print(f"  {stem}: MAE={metrics.get('MAE', 0):.4f}"
              f"  SSIM={metrics.get('SSIM', 0):.4f}"
              f"  PSNR={metrics.get('PSNR', 0):.2f}dB")

    # Average metrics
    if all_metrics:
        avg = {k: float(np.mean([m[k] for m in all_metrics if k in m]))
               for k in all_metrics[0]}
        print(f"\n[AVG] MAE={avg.get('MAE',0):.4f}  "
              f"SSIM={avg.get('SSIM',0):.4f}  PSNR={avg.get('PSNR',0):.2f}dB")
    print(f"[BATCH] Done. Results in {output_dir}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pix2Pix Inference")
    parser.add_argument('--model_dir',  default='saved_models/best_generator')
    # Single image mode
    parser.add_argument('--input',      default=None, help='Path to single satellite image')
    parser.add_argument('--output',     default='output_map.png')
    # Batch mode
    parser.add_argument('--input_dir',  default=None, help='Dir with multiple images')
    parser.add_argument('--output_dir', default='outputs/predictions')
    # Options
    parser.add_argument('--compare',    action='store_true',
                        help='Save side-by-side comparison instead of just output')
    args = parser.parse_args()

    generator = load_generator(args.model_dir)

    if args.input:
        run_single(generator, args.input, args.output, compare=args.compare)
    elif args.input_dir:
        run_batch(generator, args.input_dir, args.output_dir, compare=args.compare)
    else:
        parser.error("Provide --input (single image) or --input_dir (batch mode)")


if __name__ == '__main__':
    main()