"""Video preprocessing for vision-language models."""

import base64
import math
import os
import tempfile
import time
from io import BytesIO

import numpy as np
from PIL import Image

from razordl.core.base import logging

logger = logging.getLogger(__name__)

IMAGE_FACTOR = 28
VIDEO_MIN_PIXELS = 32 * 28 * 28
VIDEO_MAX_PIXELS = 336 * 28 * 28
MAX_RATIO = 200
VIDEO_TOTAL_PIXELS = int(float(os.environ.get("VIDEO_MAX_PIXELS", 128000 * 28 * 28 * 0.9)))
FRAME_FACTOR = 2


def _round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def _ceil_by_factor(number: int | float, factor: int) -> int:
    return math.ceil(number / factor) * factor


def _floor_by_factor(number: int | float, factor: int) -> int:
    return math.floor(number / factor) * factor


def _smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = VIDEO_MIN_PIXELS,
    max_pixels: int = VIDEO_MAX_PIXELS,
) -> tuple[int, int]:
    h_bar = max(factor, _round_by_factor(height, factor))
    w_bar = max(factor, _round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = _floor_by_factor(height / beta, factor)
        w_bar = _floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = _ceil_by_factor(height * beta, factor)
        w_bar = _ceil_by_factor(width * beta, factor)

    if max(h_bar, w_bar) / min(h_bar, w_bar) > MAX_RATIO:
        logger.warning(
            f"Absolute aspect ratio must be smaller than {MAX_RATIO}, "
            f"got {max(h_bar, w_bar) / min(h_bar, w_bar)}"
        )
        if h_bar > w_bar:
            h_bar = w_bar * MAX_RATIO
        else:
            w_bar = h_bar * MAX_RATIO
    return h_bar, w_bar


def _read_video_decord(video_bytes: bytes, nframes: int = None, sample_fps: float = None):
    """Decode video bytes via decord.  Returns (frames_array, sample_fps, ...)."""
    try:
        from decord import VideoReader, cpu
    except ImportError as e:
        raise ImportError(
            "decord is required for video preprocessing. "
            "Install with: pip install decord"
        ) from e

    try:
        buffer = BytesIO(video_bytes)
        vr = VideoReader(buffer, ctx=cpu(0))
        total_frames, video_fps = len(vr), vr.get_avg_fps()
        duration = total_frames / video_fps

        if nframes is not None:
            sample_fps = nframes / max(total_frames, 1e-6) * video_fps
            sample_fps = min(sample_fps, video_fps)
        elif sample_fps is not None:
            nframes = int(sample_fps * duration)
            nframes = min(nframes, total_frames)
        else:
            nframes = total_frames
            sample_fps = video_fps

        idx = np.linspace(0, total_frames - 1, nframes).round().astype(int).tolist()
        video = vr.get_batch(idx).asnumpy()  # (T, H, W, C) RGB
    except Exception as e:
        logger.error(f"Error reading video: {e}")
        return None, None, None, None, None

    return video, sample_fps, duration, total_frames, video_fps


def _gen_video_qwen_vl_utils(video_bytes: bytes, nframes: int = None, sample_fps: float = None):
    """Sample frames from video bytes and resize for Qwen-VL processor."""
    video, sample_fps, duration, total_frames, video_fps = _read_video_decord(
        video_bytes, nframes=nframes, sample_fps=sample_fps
    )
    if video is None:
        return None

    nframes, height, width, _ = video.shape
    min_pixels = VIDEO_MIN_PIXELS
    total_pixels = VIDEO_TOTAL_PIXELS
    max_pixels = max(
        min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR),
        int(min_pixels * 1.05),
    )

    resized_h, resized_w = _smart_resize(
        height, width,
        factor=IMAGE_FACTOR,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    try:
        import cv2
    except ImportError as e:
        raise ImportError(
            "opencv-python is required for video frame resizing. "
            "Install with: pip install opencv-python"
        ) from e

    resized = np.empty((nframes, resized_h, resized_w, 3), dtype=np.uint8)
    for i in range(nframes):
        resized[i] = cv2.resize(video[i], (resized_w, resized_h), interpolation=cv2.INTER_CUBIC)

    return {
        "video_array": resized,
        "video_width": width,
        "video_height": height,
        "resized_width": resized_w,
        "resized_height": resized_h,
        "fps": sample_fps,
        "duration": duration,
        "total_frames": total_frames,
        "video_fps": video_fps,
    }


def _image_to_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def video_path_to_base64(video_path: str, nframes: int = None, sample_fps: float = None) -> str:
    """Stitch sampled frames into an MP4 and return a single base64 string."""
    with open(video_path, "rb") as f:
        video_bytes = f.read()
    output_dict = _gen_video_qwen_vl_utils(video_bytes, nframes=nframes, sample_fps=sample_fps)

    if output_dict is None:
        return base64.b64encode(video_bytes).decode("utf-8")

    frames = output_dict["video_array"]
    h = output_dict["resized_height"]
    w = output_dict["resized_width"]
    target_fps = sample_fps if sample_fps is not None else (output_dict.get("fps") or 8.0)

    import cv2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_path = tmp.name
    tmp.close()

    writer = cv2.VideoWriter(tmp_path, fourcc, float(target_fps), (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()

    with open(tmp_path, "rb") as vf:
        base64_str = base64.b64encode(vf.read()).decode("utf-8")
    os.remove(tmp_path)
    return base64_str


def video_path_to_base64_image_lst(video_path: str, nframes: int = None, sample_fps: float = None) -> list[str]:
    """Sample video frames and return a list of base64-encoded PNG images."""
    with open(video_path, "rb") as f:
        video_bytes = f.read()
    output_dict = _gen_video_qwen_vl_utils(video_bytes, nframes=nframes, sample_fps=sample_fps)
    if output_dict is None:
        return []
    return [_image_to_base64(Image.fromarray(f)) for f in output_dict["video_array"]]
