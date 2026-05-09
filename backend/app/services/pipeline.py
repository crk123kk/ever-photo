import logging
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from app.core.config import (
    DEVICE,
    GFPGAN_WEIGHT,
    OUTPUT_DIR,
    REALESRGAN_SCALE,
    REALESRGAN_WEIGHT,
    SCRATCH_KERNEL_SIZE,
    SCRATCH_THRESHOLD,
    WEIGHTS_DIR,
)

logger = logging.getLogger(__name__)

# HuggingFace sources (accessible via hf-mirror.com)
HF_WEIGHTS = {
    "GFPGANv1.4.pth": ("th3w33knd/GFPGANv1.4", "GFPGANv1.4.pth"),
}


def _ensure_weight(path: Path) -> Path:
    if path.exists():
        return path
    name = path.name
    if name not in HF_WEIGHTS:
        raise FileNotFoundError(
            f"Weight {name} not found at {path}. "
            f"Run: python download_weights.py"
        )
    repo_id, filename = HF_WEIGHTS[name]
    logger.info("Downloading %s from HuggingFace ...", name)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from huggingface_hub import hf_hub_download

    hf_hub_download(repo_id, filename, local_dir=str(WEIGHTS_DIR))
    if not path.exists():
        raise FileNotFoundError(f"Download failed for {name}")
    return path


class RestorePipeline:
    def __init__(self):
        self.device = torch.device(DEVICE)
        self._lama = None
        self._face_restorer = None
        self._upsampler = None

    # ---- lazy loaders ----

    def _load_lama(self):
        if self._lama is not None:
            return
        from simple_lama_inpainting import SimpleLama

        self._lama = SimpleLama()

    def _load_face_restorer(self):
        if self._face_restorer is not None:
            return
        from gfpgan import GFPGANer

        _ensure_weight(GFPGAN_WEIGHT)
        self._face_restorer = GFPGANer(
            model_path=str(GFPGAN_WEIGHT),
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            device=self.device,
        )

    def _load_upsampler(self):
        if self._upsampler is not None:
            return
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer

            if not REALESRGAN_WEIGHT.exists():
                raise FileNotFoundError(f"{REALESRGAN_WEIGHT} not found")

            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=REALESRGAN_SCALE,
            )
            self._upsampler = RealESRGANer(
                scale=REALESRGAN_SCALE,
                model_path=str(REALESRGAN_WEIGHT),
                model=model,
                tile=400,
                tile_pad=10,
                pre_pad=0,
                half=self.device.type == "cuda",
            )
        except Exception as e:
            logger.warning("Real-ESRGAN unavailable (%s), will use cv2.resize", e)

    # ---- processing steps ----

    @staticmethod
    def detect_scratches(img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (SCRATCH_KERNEL_SIZE, SCRATCH_KERNEL_SIZE)
        )
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        combined = cv2.add(tophat, blackhat)
        _, mask = cv2.threshold(
            combined, SCRATCH_THRESHOLD, 255, cv2.THRESH_BINARY
        )
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.dilate(mask, dilate_kernel, iterations=1)
        return mask

    def inpaint_scratches(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        try:
            self._load_lama()
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            mask_pil = Image.fromarray(mask).convert("L")
            result = self._lama(img_pil, mask_pil)
            return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.warning("LaMa unavailable (%s), using OpenCV inpainting", e)
            return cv2.inpaint(img, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

    def restore_faces(self, img: np.ndarray) -> np.ndarray:
        try:
            self._load_face_restorer()
            _, _, output = self._face_restorer.enhance(
                img, has_aligned=False, only_center_face=False, paste_back=True
            )
            if output is not None:
                return output
        except Exception as e:
            logger.warning("Face restoration failed (%s), using original", e)
        return img

    def upscale(self, img: np.ndarray) -> np.ndarray:
        try:
            self._load_upsampler()
            if self._upsampler is not None:
                output, _ = self._upsampler.enhance(img, outscale=REALESRGAN_SCALE)
                return output
        except Exception as e:
            logger.warning("Real-ESRGAN failed (%s), using cv2.resize", e)
        h, w = img.shape[:2]
        return cv2.resize(
            img, (w * REALESRGAN_SCALE, h * REALESRGAN_SCALE),
            interpolation=cv2.INTER_LANCZOS4,
        )

    # ---- main entry ----

    def restore(self, input_path: str, output_path: str) -> str:
        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Cannot read image: {input_path}")

        logger.info("Step 1/3: Scratch detection + inpainting")
        mask = self.detect_scratches(img)
        scratch_ratio = mask.sum() / (mask.shape[0] * mask.shape[1]) / 255
        if 0.001 < scratch_ratio < 0.3:
            logger.info("Scratches detected (%.1f%%), running inpainting", scratch_ratio * 100)
            img = self.inpaint_scratches(img, mask)
        else:
            logger.info("No significant scratches detected, skipping inpainting")

        logger.info("Step 2/3: Face restoration")
        img = self.restore_faces(img)

        logger.info("Step 3/3: Super resolution (%dx)", REALESRGAN_SCALE)
        img = self.upscale(img)

        cv2.imwrite(output_path, img)
        logger.info("Done: %s", output_path)
        return output_path


pipeline = RestorePipeline()
