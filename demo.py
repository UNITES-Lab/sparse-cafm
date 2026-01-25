"""
SparseC-AFM: AFM Super-Resolution Demo Application

A simple Gradio-based web app for experimenting with Swin Transformer
models for AFM (Atomic Force Microscopy) map super-resolution.

Usage:
    python demo.py

Then open http://127.0.0.1:7860 in your browser.
"""

import io
import tempfile
from pathlib import Path
from typing import Tuple, Optional

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy import stats as scipy_stats

# Add src to path for model imports
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.models.our_method.swin_cafm import SwinCAFM


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

MODEL_CONFIGS = {
    "2x": {"input_size": 64, "upscale": 2, "weights": "data/weights/2x/2x.pth"},
    "4x": {"input_size": 64, "upscale": 4, "weights": "data/weights/4x/4x.pth"},
    "8x": {"input_size": 32, "upscale": 8, "weights": "data/weights/8x/8x.pth"},
}

# Demo samples (center-cropped for fast processing)
DEMO_SAMPLES = {
    "MoS2 on SiO2 - Topography": "demo/MoS2_SiO2_Topography.npy",
    "MoS2 on SiO2 - Current": "demo/MoS2_SiO2_Current.npy",
    "MoS2 on Sapphire - Topography": "demo/MoS2_Sapphire_Topography.npy",
    "MoS2 on Sapphire - Current": "demo/MoS2_Sapphire_Current.npy",
}

COLORMAPS = ["viridis", "plasma", "inferno", "magma", "cividis", "hot", "coolwarm", "gray"]

SUPPORTED_FORMATS = {
    ".npy": "NumPy array",
    ".tif": "TIFF image",
    ".tiff": "TIFF image",
    ".png": "PNG image",
    ".jpg": "JPEG image",
    ".jpeg": "JPEG image",
    ".bmp": "BMP image",
    ".webp": "WebP image",
}


# ─────────────────────────────────────────────────────────────────────────────
# Device Detection
# ─────────────────────────────────────────────────────────────────────────────

def get_available_devices() -> list[str]:
    """Detect available compute devices."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        devices.append(f"cuda ({gpu_name})")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append("mps (Apple Silicon)")
    return devices


def parse_device(device_str: str) -> str:
    """Extract device name from display string."""
    if device_str.startswith("cuda"):
        return "cuda"
    elif device_str.startswith("mps"):
        return "mps"
    return "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# Image I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_map(filepath: str) -> np.ndarray:
    """
    Load a conductivity or topology map from various formats.
    Returns a 2D numpy array (grayscale).
    """
    ext = Path(filepath).suffix.lower()

    if ext == ".npy":
        data = np.load(filepath)
        # Handle 3D arrays (take first channel or squeeze)
        if data.ndim == 3:
            data = data[:, :, 0] if data.shape[2] <= 4 else data[0]
        return data.astype(np.float32)

    elif ext in [".tif", ".tiff"]:
        try:
            import tifffile
            data = tifffile.imread(filepath)
        except ImportError:
            # Fallback to PIL
            img = Image.open(filepath)
            data = np.array(img)
        if data.ndim == 3:
            data = data[:, :, 0]
        return data.astype(np.float32)

    elif ext in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]:
        img = Image.open(filepath).convert("L")  # Convert to grayscale
        return np.array(img, dtype=np.float32)

    else:
        raise ValueError(f"Unsupported format: {ext}. Supported: {list(SUPPORTED_FORMATS.keys())}")


def apply_colormap(data: np.ndarray, cmap_name: str = "viridis") -> np.ndarray:
    """Apply a matplotlib colormap to grayscale data, returning RGB uint8."""
    # Normalize to [0, 1]
    normalized = (data - data.min()) / (data.max() - data.min() + 1e-8)

    # Apply colormap
    cmap = plt.get_cmap(cmap_name)
    colored = cmap(normalized)[:, :, :3]  # Drop alpha channel

    return (colored * 255).astype(np.uint8)


def save_to_format(data: np.ndarray, format: str, cmap_name: str = "viridis") -> str:
    """Save array to a temporary file in the specified format."""
    temp_dir = tempfile.gettempdir()

    if format == "npy":
        filepath = Path(temp_dir) / "upsampled_result.npy"
        np.save(filepath, data)

    elif format == "tiff":
        filepath = Path(temp_dir) / "upsampled_result.tiff"
        try:
            import tifffile
            tifffile.imwrite(filepath, data.astype(np.float32))
        except ImportError:
            # Fallback: save as 16-bit normalized
            normalized = (data - data.min()) / (data.max() - data.min() + 1e-8)
            img = Image.fromarray((normalized * 65535).astype(np.uint16))
            img.save(filepath)

    elif format == "png":
        filepath = Path(temp_dir) / "upsampled_result.png"
        colored = apply_colormap(data, cmap_name)
        Image.fromarray(colored).save(filepath)

    elif format == "csv":
        filepath = Path(temp_dir) / "upsampled_result.csv"
        np.savetxt(filepath, data, delimiter=",")

    else:
        raise ValueError(f"Unsupported export format: {format}")

    return str(filepath)


# ─────────────────────────────────────────────────────────────────────────────
# Model Management
# ─────────────────────────────────────────────────────────────────────────────

# Global model cache: {(scale, device): model}
_MODEL_CACHE: dict[Tuple[str, str], torch.nn.Module] = {}


def create_model(scale: str) -> torch.nn.Module:
    """Create model architecture for the given scale."""
    config = MODEL_CONFIGS[scale]
    upscale = config["upscale"]
    img_size = config["input_size"]

    return SwinCAFM(
        upscale=upscale,
        img_size=img_size,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        drop_path_rate=0.1,
        norm_layer=torch.nn.LayerNorm,
        upsampler="pixelshuffle",
        resi_connection="1conv",
    )


def get_model(scale: str, device: str) -> torch.nn.Module:
    """Load and cache a model for the given scale and device."""
    key = (scale, device)

    if key not in _MODEL_CACHE:
        config = MODEL_CONFIGS[scale]
        weights_path = Path(__file__).parent / config["weights"]

        if not weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {weights_path}")

        # Load weights file
        loaded = torch.load(weights_path, map_location=device, weights_only=False)

        # Handle different save formats:
        # 1. Full model object (SwinCAFM) - use directly
        # 2. State dict (OrderedDict) - load into new model
        # 3. Dict with "params" key - extract and load
        if isinstance(loaded, SwinCAFM):
            model = loaded
        else:
            model = create_model(scale)
            state_dict = loaded
            if isinstance(state_dict, dict) and "params" in state_dict:
                state_dict = state_dict["params"]
            model.load_state_dict(state_dict, strict=False)

        model = model.to(device).eval()
        _MODEL_CACHE[key] = model

    return _MODEL_CACHE[key]


def clear_model_cache():
    """Clear the model cache to free memory."""
    global _MODEL_CACHE
    _MODEL_CACHE.clear()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ─────────────────────────────────────────────────────────────────────────────
# Image Processing (Tiled)
# ─────────────────────────────────────────────────────────────────────────────

def pad_to_multiple(data: np.ndarray, tile_size: int) -> Tuple[np.ndarray, Tuple[int, int], str]:
    """
    Pad image so dimensions are multiples of tile_size.
    Returns (padded_data, original_shape, warning_message).
    """
    h, w = data.shape[:2]
    original_shape = (h, w)
    warnings = []

    # Calculate padding needed
    pad_h = (tile_size - h % tile_size) % tile_size
    pad_w = (tile_size - w % tile_size) % tile_size

    if pad_h > 0 or pad_w > 0:
        warnings.append(f"Input ({h}x{w}) padded to ({h + pad_h}x{w + pad_w}) for tiling.")
        data = np.pad(
            data,
            ((0, pad_h), (0, pad_w)),
            mode='reflect'  # Use reflect padding to avoid edge artifacts
        )

    warning = " ".join(warnings)
    return data, original_shape, warning


def process_tiled(
    data: np.ndarray,
    model: torch.nn.Module,
    tile_size: int,
    upscale: int,
    device: str,
) -> np.ndarray:
    """
    Process a large image by splitting into tiles, upsampling each, and stitching.

    Args:
        data: Input image (H, W), normalized to [0, 1]
        model: The upsampling model
        tile_size: Size of each tile (e.g., 64 for 2x/4x models)
        upscale: Upscaling factor (2, 4, or 8)
        device: Compute device

    Returns:
        Upsampled image (H*upscale, W*upscale)
    """
    h, w = data.shape
    out_h, out_w = h * upscale, w * upscale

    # Initialize output array
    output = np.zeros((out_h, out_w), dtype=np.float32)

    # Process each tile
    n_tiles_h = h // tile_size
    n_tiles_w = w // tile_size

    for i in range(n_tiles_h):
        for j in range(n_tiles_w):
            # Extract tile
            y_start = i * tile_size
            x_start = j * tile_size
            tile = data[y_start:y_start + tile_size, x_start:x_start + tile_size]

            # Run inference on tile
            X = torch.tensor(tile, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                tile_out = model(X).cpu().numpy()[0]

            # Place in output
            out_y = i * tile_size * upscale
            out_x = j * tile_size * upscale
            output[out_y:out_y + tile_size * upscale, out_x:out_x + tile_size * upscale] = tile_out

    return output


def center_crop(data: np.ndarray, target_size: int) -> np.ndarray:
    """
    Center crop the input to target_size x target_size.
    Used for demo samples for fast processing.
    """
    h, w = data.shape[:2]
    start_h = (h - target_size) // 2
    start_w = (w - target_size) // 2
    return data[start_h:start_h + target_size, start_w:start_w + target_size]


# ─────────────────────────────────────────────────────────────────────────────
# Statistics (Gwyddion-inspired)
# ─────────────────────────────────────────────────────────────────────────────

def compute_statistics(arr: np.ndarray) -> dict:
    """Compute Gwyddion-inspired surface statistics."""
    flat = arr.flatten()
    centered = arr - np.mean(arr)

    return {
        "Dimensions": f"{arr.shape[0]} x {arr.shape[1]} px",
        "Min": f"{arr.min():.6g}",
        "Max": f"{arr.max():.6g}",
        "Mean": f"{arr.mean():.6g}",
        "Median": f"{np.median(arr):.6g}",
        "Std Dev (σ)": f"{arr.std():.6g}",
        "RMS Roughness (Rq)": f"{np.sqrt(np.mean(centered**2)):.6g}",
        "Avg Roughness (Ra)": f"{np.mean(np.abs(centered)):.6g}",
        "Peak-to-Valley (Rz)": f"{arr.max() - arr.min():.6g}",
        "Skewness": f"{scipy_stats.skew(flat):.4f}",
        "Kurtosis": f"{scipy_stats.kurtosis(flat):.4f}",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main Inference Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(
    file,
    demo_sample: str,
    scale: str,
    device_str: str,
    colormap: str,
    export_format: str,
) -> Tuple[Optional[Tuple[np.ndarray, np.ndarray]], dict, Optional[str], str]:
    """
    Main inference function.

    Returns:
        - (input_image, output_image) tuple for ImageSlider
        - statistics dictionary
        - path to downloadable file
        - status/warning message
    """
    # Determine input source: demo sample or uploaded file
    use_demo = demo_sample and demo_sample != "Upload your own"

    if not use_demo and file is None:
        return None, {}, None, "Please select a demo sample or upload an image file."

    try:
        # Load input
        if use_demo:
            demo_path = Path(__file__).parent / DEMO_SAMPLES[demo_sample]
            data = np.load(demo_path)
        else:
            data = load_map(file.name)

        original_shape = data.shape
        original_min, original_max = data.min(), data.max()

        # Get model config
        config = MODEL_CONFIGS[scale]
        tile_size = config["input_size"]
        upscale_factor = config["upscale"]

        # Normalize to [0, 1]
        normalized = (data - original_min) / (original_max - original_min + 1e-8)

        # Load model
        device = parse_device(device_str)
        model = get_model(scale, device)

        if use_demo:
            # Demo samples: use center crop for fast processing
            cropped = center_crop(normalized, tile_size)

            # Single tile inference
            X = torch.tensor(cropped, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(X).cpu().numpy()[0]

            input_vis = cropped
            out_h, out_w = output.shape
            status = f"Demo: Center-cropped {original_shape[0]}x{original_shape[1]} to {tile_size}x{tile_size} -> {out_h}x{out_w} using {scale} model on {device}."

        else:
            # User uploads: full tiled processing (preserves all pixels)
            padded, orig_shape, pad_warning = pad_to_multiple(normalized, tile_size)
            padded_h, padded_w = padded.shape

            # Process using tiled approach
            output = process_tiled(padded, model, tile_size, upscale_factor, device)

            # Crop output back to original size (scaled)
            out_h = orig_shape[0] * upscale_factor
            out_w = orig_shape[1] * upscale_factor
            output = output[:out_h, :out_w]

            # Input visualization matches original
            input_vis = normalized[:orig_shape[0], :orig_shape[1]]

            # Build status message
            n_tiles = (padded_h // tile_size) * (padded_w // tile_size)
            status = f"Processed {original_shape[0]}x{original_shape[1]} in {n_tiles} tiles -> {out_h}x{out_w} using {scale} model on {device}."
            if pad_warning:
                status = f"Note: {pad_warning}\n{status}"

        # Denormalize output to original scale
        output_denorm = output * (original_max - original_min) + original_min

        # Apply colormap for visualization
        input_colored = apply_colormap(input_vis, colormap)
        output_colored = apply_colormap(output, colormap)

        # Compute statistics on denormalized output
        stats = compute_statistics(output_denorm)

        # Save to requested format
        download_path = save_to_format(output_denorm, export_format, colormap)

        return (input_colored, output_colored), stats, download_path, status

    except Exception as e:
        return None, {}, None, f"Error: {str(e)}"


# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────────────────────

def create_app() -> gr.Blocks:
    """Create and configure the Gradio application."""

    with gr.Blocks(title="SparseC-AFM: AFM Super-Resolution") as app:

        gr.Markdown("""
        # SparseC-AFM: AFM Super-Resolution

        **Supported formats:** .npy, .tiff, .png, .jpg, .bmp, .webp
        """)

        with gr.Row():
            # Left column: inputs
            with gr.Column(scale=1):
                # Demo sample selector
                demo_dropdown = gr.Dropdown(
                    choices=["Upload your own"] + list(DEMO_SAMPLES.keys()),
                    value="Upload your own",
                    label="Select",
                )

                file_input = gr.File(
                    label="Or Upload Your Own (full resolution)",
                    file_types=[".npy", ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp"],
                )

                with gr.Row():
                    scale_dropdown = gr.Dropdown(
                        choices=list(MODEL_CONFIGS.keys()),
                        value="4x",
                        label="Upscale Factor",
                    )
                    device_dropdown = gr.Dropdown(
                        choices=get_available_devices(),
                        value=get_available_devices()[0],
                        label="Compute Device",
                    )

                with gr.Row():
                    colormap_dropdown = gr.Dropdown(
                        choices=COLORMAPS,
                        value="viridis",
                        label="Colormap",
                    )
                    export_dropdown = gr.Dropdown(
                        choices=["npy", "tiff", "png", "csv"],
                        value="npy",
                        label="Download Format",
                    )

                run_button = gr.Button("Upsample", variant="primary", size="lg")

                status_box = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=2,
                )

            # Right column: outputs
            with gr.Column(scale=2):
                # Image comparison slider
                image_slider = gr.ImageSlider(
                    label="Original (left) vs Upsampled (right)",
                    type="numpy",
                )

                with gr.Row():
                    # Statistics panel
                    stats_output = gr.JSON(
                        label="Sample Statistics",
                    )

                    # Download
                    download_output = gr.File(
                        label="Download Result",
                    )

        # Connect the interface
        run_button.click(
            fn=run_inference,
            inputs=[file_input, demo_dropdown, scale_dropdown, device_dropdown, colormap_dropdown, export_dropdown],
            outputs=[image_slider, stats_output, download_output, status_box],
        )

    return app


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
    )
