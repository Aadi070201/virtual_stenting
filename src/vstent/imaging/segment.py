"""Vessel segmentation: shared pre-processing, augmentation and metrics (no torch needed).

``preprocess_image`` is the single pre-processing path for training, ONNX inference and the
app: greyscale image → resized to the model size → [0, 1] → standardised with fixed constants.
Probabilities are mapped back to the original image size with ``probability_to_image``.
"""

import cv2
import numpy as np

from vstent.config import StrictModel

IMAGE_SIZE = 512
# Mean / std of the ImageNet RGB statistics averaged over channels: smp sums the pretrained
# first-conv weights over RGB for in_channels=1, so a grey input should be scaled like one
# ImageNet channel.
PIXEL_MEAN = 0.449
PIXEL_STD = 0.226
THRESHOLD = 0.5


def to_unit(image: np.ndarray) -> np.ndarray:
    """Greyscale image (uint8 0–255, uint16 0–65535 or float in [0, 1]) → float32 in [0, 1]."""
    img = np.asarray(image)
    if img.ndim == 3:  # colour PNG / JPEG of an angiogram: take the luminance
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY if img.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    return np.clip(img.astype(np.float32), 0.0, 1.0)


def resize(image: np.ndarray, size: int, interpolation: int = cv2.INTER_AREA) -> np.ndarray:
    """Square resize to ``size`` × ``size`` (no-op if already that size)."""
    if image.shape[:2] == (size, size):
        return image
    return cv2.resize(image, (size, size), interpolation=interpolation)


def standardise(unit: np.ndarray) -> np.ndarray:
    """[0, 1] image → model input values (float32)."""
    return ((unit - PIXEL_MEAN) / PIXEL_STD).astype(np.float32)


def preprocess_image(image: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """Model input (1, size, size) float32 from a greyscale image of any size."""
    return standardise(resize(to_unit(image), size))[None]


def probability_to_image(prob: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Model-size probability map (size, size) → original image ``shape`` (H, W), float32."""
    if prob.shape == tuple(shape):
        return prob.astype(np.float32)
    return cv2.resize(prob.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


class AugmentConfig(StrictModel):
    """Training augmentation ranges (no flips: angiogram orientation is meaningful)."""

    rotation_deg: float = 15.0
    scale: tuple[float, float] = (0.9, 1.1)
    brightness: float = 0.1  # additive, in [0, 1] intensity units
    contrast: tuple[float, float] = (0.8, 1.2)
    gamma: tuple[float, float] = (0.8, 1.25)
    noise_std: tuple[float, float] = (0.0, 0.03)
    p_noise: float = 0.5
    p_clahe: float = 0.3
    clahe_clip: float = 2.0


def augment(
    image: np.ndarray, mask: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Random rotation + scale (one affine warp), optional CLAHE, brightness / contrast /
    gamma and Gaussian noise.

    Args:
        image: uint8 greyscale (H, W).
        mask: uint8 {0, 1} (H, W).

    Returns:
        Augmented image as float32 in [0, 1] and mask as uint8 {0, 1}.
    """
    cfg = cfg or AugmentConfig()
    h, w = image.shape
    angle = rng.uniform(-cfg.rotation_deg, cfg.rotation_deg)
    scale = rng.uniform(*cfg.scale)
    m = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), angle, scale)
    border = cv2.BORDER_REFLECT_101
    img = cv2.warpAffine(image, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=border)
    msk = cv2.warpAffine(mask, m, (w, h), flags=cv2.INTER_NEAREST, borderMode=border)
    if rng.random() < cfg.p_clahe:
        img = cv2.createCLAHE(clipLimit=cfg.clahe_clip, tileGridSize=(8, 8)).apply(img)
    x = img.astype(np.float32) / 255.0
    x = x * rng.uniform(*cfg.contrast) + rng.uniform(-cfg.brightness, cfg.brightness)
    x = np.clip(x, 0.0, 1.0) ** rng.uniform(*cfg.gamma)
    if rng.random() < cfg.p_noise:
        x = x + rng.normal(0.0, rng.uniform(*cfg.noise_std), size=x.shape).astype(np.float32)
    return np.clip(x, 0.0, 1.0).astype(np.float32), msk


def dice_score(pred: np.ndarray, target: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Per-image Dice of binary masks with shape (N, H, W) (or one (H, W) mask).

    An image with empty prediction and empty target scores 1.
    """
    p = np.asarray(pred, dtype=bool).reshape(-1, *np.shape(pred)[-2:])
    t = np.asarray(target, dtype=bool).reshape(p.shape)
    inter = (p & t).sum(axis=(1, 2))
    total = p.sum(axis=(1, 2)) + t.sum(axis=(1, 2))
    return np.where(total == 0, 1.0, 2.0 * inter / (total + eps))
