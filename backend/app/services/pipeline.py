import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch
from PIL import Image

from app.core.config import (
    CODEFORMER_WEIGHT,
    DEVICE,
    GFPGAN_WEIGHT,
    OUTPUT_DIR,
    REALESRGAN_WEIGHT,
    SCRATCH_KERNEL_SIZE,
    SCRATCH_THRESHOLD,
    WEIGHTS_DIR,
)

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, float, str], None]

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


@dataclass
class RestoreParams:
    scratch_enabled: bool = True
    scratch_threshold: int = SCRATCH_THRESHOLD
    scratch_kernel_size: int = SCRATCH_KERNEL_SIZE
    face_enabled: bool = True
    face_model: str = "gfpgan"  # "gfpgan" or "codeformer"
    fidelity_weight: float = 0.5
    upscale_enabled: bool = True
    upscale_factor: int = 2


class RestorePipeline:
    def __init__(self):
        self.device = torch.device(DEVICE)
        self._lama = None
        self._gfpgan = None
        self._codeformer = None
        self._upsampler = None

    # ---- lazy loaders ----

    def _load_lama(self):
        if self._lama is not None:
            return
        from simple_lama_inpainting import SimpleLama

        self._lama = SimpleLama()

    def _load_gfpgan(self):
        if self._gfpgan is not None:
            return
        try:
            from gfpgan import GFPGANer

            _ensure_weight(GFPGAN_WEIGHT)
            self._gfpgan = GFPGANer(
                model_path=str(GFPGAN_WEIGHT),
                upscale=1,
                arch="clean",
                channel_multiplier=2,
                device=self.device,
            )
            logger.info("GFPGAN loaded from %s", GFPGAN_WEIGHT)
        except Exception as e:
            logger.warning("GFPGAN unavailable (%s)", e)

    def _load_codeformer(self):
        if self._codeformer is not None:
            return
        try:
            from codeformer.basicsr.archs.codeformer_arch import CodeFormer as CodeFormerArch

            if not CODEFORMER_WEIGHT.exists():
                raise FileNotFoundError(f"{CODEFORMER_WEIGHT} not found")

            codeformer = CodeFormerArch(
                dim_embd=512,
                codebook_size=1024,
                n_head=8,
                n_layers=9,
                connect_list=["32", "64", "128", "256"],
            ).to(self.device)

            ckpt = torch.load(str(CODEFORMER_WEIGHT), map_location=self.device, weights_only=False)
            if "params_ema" in ckpt:
                codeformer.load_state_dict(ckpt["params_ema"])
            elif "params" in ckpt:
                codeformer.load_state_dict(ckpt["params"])
            else:
                codeformer.load_state_dict(ckpt)
            codeformer.eval()

            self._codeformer = codeformer
            logger.info("CodeFormer loaded from %s", CODEFORMER_WEIGHT)
        except Exception as e:
            logger.warning("CodeFormer unavailable (%s)", e)

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
                scale=2,
            )
            self._upsampler = RealESRGANer(
                scale=2,
                model_path=str(REALESRGAN_WEIGHT),
                model=model,
                tile=400,
                tile_pad=10,
                pre_pad=0,
                half=self.device.type == "cuda",
            )
            logger.info("Real-ESRGAN loaded from %s", REALESRGAN_WEIGHT)
        except Exception as e:
            logger.warning("Real-ESRGAN unavailable (%s), will use cv2.resize", e)

    # ---- processing steps ----

    @staticmethod
    def detect_scratches(
        img: np.ndarray,
        threshold: int = SCRATCH_THRESHOLD,
        kernel_size: int = SCRATCH_KERNEL_SIZE,
    ) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (kernel_size, kernel_size)
        )
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        combined = cv2.add(tophat, blackhat)
        _, mask = cv2.threshold(combined, threshold, 255, cv2.THRESH_BINARY)
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

    def restore_faces(self, img: np.ndarray, face_model: str = "gfpgan", fidelity_weight: float = 0.5) -> np.ndarray:
        if face_model == "codeformer":
            result = self._restore_faces_codeformer(img, fidelity_weight)
            if result is not None:
                return result
            # Fallback to GFPGAN if CodeFormer failed
            logger.info("CodeFormer failed or unavailable, falling back to GFPGAN")

        return self._restore_faces_gfpgan(img)

    def _restore_faces_gfpgan(self, img: np.ndarray) -> np.ndarray:
        try:
            self._load_gfpgan()
            if self._gfpgan is None:
                return img
            _, _, output = self._gfpgan.enhance(
                img, has_aligned=False, only_center_face=False, paste_back=True
            )
            if output is not None:
                return output
        except Exception as e:
            logger.warning("GFPGAN face restoration failed (%s)", e)
        return img

    def _restore_faces_codeformer(self, img: np.ndarray, fidelity_weight: float = 0.5) -> Optional[np.ndarray]:
        try:
            self._load_codeformer()
            if self._codeformer is None:
                return None

            from codeformer.facelib.utils.face_restoration_helper import FaceRestoreHelper

            face_helper = FaceRestoreHelper(
                upscale_factor=1,
                face_size=512,
                crop_ratio=(1, 1),
                det_model="retinaface_resnet50",
                save_ext="png",
                use_parse=True,
                device=self.device,
            )
            face_helper.clean_all()
            face_helper.read_image(img)
            face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
            face_helper.align_warp_face()

            for cropped_face in face_helper.cropped_faces:
                cropped_face_t = torch.from_numpy(cropped_face).permute(2, 0, 1).float() / 255.0
                cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)
                with torch.no_grad():
                    output = self._codeformer(cropped_face_t, w=fidelity_weight)[0]
                    output = (output.squeeze(0).permute(1, 2, 0) * 255.0).cpu().numpy()
                output = output.clip(0, 255).astype("uint8")
                face_helper.add_restored_face(output)

            face_helper.get_inverse_affine(None)
            restored_img = face_helper.paste_faces_to_input_image()
            face_helper.clean_all()
            return restored_img
        except Exception as e:
            logger.warning("CodeFormer face restoration failed (%s)", e)
            return None

    def upscale(self, img: np.ndarray, scale: int = 2) -> np.ndarray:
        try:
            self._load_upsampler()
            if self._upsampler is not None:
                output, _ = self._upsampler.enhance(img, outscale=scale)
                return output
        except Exception as e:
            logger.warning("Real-ESRGAN failed (%s), using cv2.resize", e)
        h, w = img.shape[:2]
        return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)

    # ---- main entry ----

    def restore(
        self,
        input_path: str,
        output_path: str,
        params: Optional[RestoreParams] = None,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> str:
        if params is None:
            params = RestoreParams()

        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Cannot read image: {input_path}")

        # Step 1: Scratch detection + inpainting
        if params.scratch_enabled:
            if progress_cb:
                progress_cb(1, 0.1, "Scratch detection")
            logger.info("Step 1/3: Scratch detection + inpainting")
            mask = self.detect_scratches(
                img,
                threshold=params.scratch_threshold,
                kernel_size=params.scratch_kernel_size,
            )
            scratch_ratio = mask.sum() / (mask.shape[0] * mask.shape[1]) / 255
            if 0.001 < scratch_ratio < 0.3:
                logger.info("Scratches detected (%.1f%%), running inpainting", scratch_ratio * 100)
                if progress_cb:
                    progress_cb(1, 0.2, "Inpainting scratches")
                img = self.inpaint_scratches(img, mask)
            else:
                logger.info("No significant scratches detected, skipping inpainting")
        else:
            logger.info("Step 1/3: Scratch detection skipped")
            if progress_cb:
                progress_cb(1, 0.33, "Scratch step skipped")

        # Step 2: Face restoration
        if params.face_enabled:
            if progress_cb:
                progress_cb(2, 0.4, "Face restoration")
            logger.info("Step 2/3: Face restoration")
            img = self.restore_faces(img, face_model=params.face_model, fidelity_weight=params.fidelity_weight)
        else:
            logger.info("Step 2/3: Face restoration skipped")
            if progress_cb:
                progress_cb(2, 0.66, "Face step skipped")

        # Step 3: Super resolution
        if params.upscale_enabled:
            if progress_cb:
                progress_cb(3, 0.7, "Super resolution")
            logger.info("Step 3/3: Super resolution (%dx)", params.upscale_factor)
            img = self.upscale(img, scale=params.upscale_factor)
        else:
            logger.info("Step 3/3: Super resolution skipped")
            if progress_cb:
                progress_cb(3, 0.95, "Upscale step skipped")

        cv2.imwrite(output_path, img)
        if progress_cb:
            progress_cb(3, 1.0, "Complete")
        logger.info("Done: %s", output_path)
        return output_path


pipeline = RestorePipeline()
