"""
Visualization helpers
- heatmapoverlay
- Bbox draw
- comparison grid
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union


# color definitions
COLORS = {
    "misalignment": (255, 100, 100),  # red
    "artifact": (100, 100, 255),      # blue
    "default": (255, 255, 0),         # yellow
}


def overlay_heatmap(
    image: Union[str, Image.Image],
    heatmap_path: str,
    alpha: float = 0.4,
    colormap: str = "jet",
) -> Image.Image:
    """
    Overlay a heatmap on the image.

    Args:
        image: source image (path or PIL Image)
        heatmap_path: heatmap path (grayscale PNG)
        alpha: overlayalpha
        colormap: color mapping (jet / hot / viridis)

    Returns:
        PIL.Image: overlayafterof image
    """
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    else:
        image = image.copy()

    # loadheatmap
    heatmap = Image.open(heatmap_path).convert("L")
    heatmap = heatmap.resize(image.size, Image.BICUBIC)
    heatmap_np = np.array(heatmap).astype(np.float32) / 255.0

    # apply colormap
    try:
        import matplotlib.cm as cm
        cmap = cm.get_cmap(colormap)
        colored = cmap(heatmap_np)[:, :, :3]  # RGB, 0-1
        colored = (colored * 255).astype(np.uint8)
    except ImportError:
        # simple red-heatmap fallback
        colored = np.zeros((*heatmap_np.shape, 3), dtype=np.uint8)
        colored[:, :, 0] = (heatmap_np * 255).astype(np.uint8)

    heatmap_rgb = Image.fromarray(colored)

    # overlay
    result = Image.blend(image, heatmap_rgb, alpha=alpha)
    return result


def draw_bboxes(
    image: Union[str, Image.Image],
    bboxes: List[Dict],
    line_width: int = 3,
    font_size: int = 14,
    show_label: bool = False,
    color: Tuple[int, int, int] = (255, 0, 0),
) -> Image.Image:
    """
    Draw bboxes on the image.

    Args:
        image: source image
        bboxes: list of bboxes; each contains box_2d, label, desc
        line_width: stroke width
        font_size: font size
        show_label: whether to show the label
        color: box color (default red)

    Returns:
        PIL.Image: annotationafterof image
    """
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    else:
        image = image.copy()

    draw = ImageDraw.Draw(image)

    # Try loading the font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except (IOError, OSError):
        font = ImageFont.load_default()

    for bbox_item in bboxes:
        if isinstance(bbox_item, dict):
            box = bbox_item.get("box_2d", [])
            label = bbox_item.get("label", "")
            desc = bbox_item.get("desc", "")
        elif isinstance(bbox_item, (list, tuple)) and len(bbox_item) == 4:
            box = bbox_item
            label = "bbox"
            desc = ""
        else:
            continue

        if len(box) != 4:
            continue

        x0, y0, x1, y1 = box

        # draw rectangle
        draw.rectangle([x0, y0, x1, y1], outline=color, width=line_width)

        # draw label
        if show_label and (label or desc):
            text = label
            if desc:
                text = f"{label}: {desc[:30]}"
            # textbackground
            text_bbox = draw.textbbox((x0, y0 - font_size - 4), text, font=font)
            draw.rectangle(text_bbox, fill=color)
            draw.text((x0, y0 - font_size - 4), text, fill=(255, 255, 255), font=font)

    return image


def create_comparison_grid(
    original: Union[str, Image.Image],
    edited: Union[str, Image.Image],
    feedback_vis: Optional[Union[str, Image.Image]] = None,
    caption: str = "",
    cell_size: int = 512,
) -> Image.Image:
    """
    Build a comparison grid (source | feedback visualization | edited).

    Args:
        original: source image
        edited: edited image
        feedback_vis: feedback visualization image (optional; e.g. with heatmap/bbox overlays)
        caption: titletext
        cell_size: per-cell size

    Returns:
        PIL.Image: comparison grid
    """
    def load_and_resize(img):
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        return img.resize((cell_size, cell_size), Image.BICUBIC)

    original = load_and_resize(original)
    edited = load_and_resize(edited)

    if feedback_vis is not None:
        feedback_vis = load_and_resize(feedback_vis)
        n_cols = 3
    else:
        n_cols = 2

    header_height = 40 if caption else 0
    label_height = 25

    width = cell_size * n_cols
    height = cell_size + header_height + label_height

    grid = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except (IOError, OSError):
        font = ImageFont.load_default()
        title_font = font

    # title
    if caption:
        truncated_caption = caption[:80] + "..." if len(caption) > 80 else caption
        draw.text((10, 10), truncated_caption, fill=(0, 0, 0), font=title_font)

    # paste image
    y_offset = header_height
    grid.paste(original, (0, y_offset))
    col = 1
    if feedback_vis is not None:
        grid.paste(feedback_vis, (cell_size, y_offset))
        col = 2
    grid.paste(edited, (cell_size * col, y_offset))

    # label
    label_y = header_height + cell_size + 2
    labels = ["Original"]
    if feedback_vis is not None:
        labels.append("Feedback")
    labels.append("Edited")

    for i, label in enumerate(labels):
        x = i * cell_size + cell_size // 2 - 30
        draw.text((x, label_y), label, fill=(0, 0, 0), font=font)

    return grid


def get_heatmap_high_regions(
    heatmap_path: str,
    threshold: float = 0.5,
    image_size: int = 512,
) -> List[str]:
    """
    Analyse a heatmap and return a textual descriptionof the highlighted regions.

    Args:
        heatmap_path: heatmap path
        threshold: highlight threshold (0-1)
        image_size: image size (used for region partitioning)

    Returns:
        region descriptionlist, e.g. ["upper-left", "center"]
    """
    if not Path(heatmap_path).exists():
        return []

    heatmap = Image.open(heatmap_path).convert("L")
    heatmap = heatmap.resize((image_size, image_size), Image.BICUBIC)
    heatmap_np = np.array(heatmap).astype(np.float32) / 255.0

    regions = []
    grid_size = image_size // 3

    region_names = [
        ["upper-left", "upper-center", "upper-right"],
        ["middle-left", "center", "middle-right"],
        ["lower-left", "lower-center", "lower-right"],
    ]

    for row in range(3):
        for col in range(3):
            y0 = row * grid_size
            y1 = (row + 1) * grid_size
            x0 = col * grid_size
            x1 = (col + 1) * grid_size
            region_val = heatmap_np[y0:y1, x0:x1].mean()
            if region_val > threshold:
                regions.append(region_names[row][col])

    return regions
