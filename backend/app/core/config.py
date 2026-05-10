import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
WEIGHTS_DIR = BASE_DIR / "weights"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
WEIGHTS_DIR.mkdir(exist_ok=True)

def _detect_device() -> str:
    import logging
    logger = logging.getLogger(__name__)
    if os.environ.get("FORCE_CPU") == "1":
        logger.info("FORCE_CPU=1, using CPU")
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            try:
                t = torch.zeros(1, device="cuda")
                del t
                logger.info("CUDA available and working, using GPU")
                return "cuda"
            except RuntimeError:
                logger.warning(
                    "CUDA reported available but tensor allocation failed "
                    "(likely missing sm_120 support for RTX 5070 Blackwell). "
                    "Falling back to CPU. Install PyTorch with cu128+ support."
                )
                return "cpu"
    except ImportError:
        pass
    return "cpu"


DEVICE = _detect_device()

# GFPGAN
GFPGAN_WEIGHT = WEIGHTS_DIR / "GFPGANv1.4.pth"

# Real-ESRGAN
REALESRGAN_SCALE = 2
REALESRGAN_WEIGHT = WEIGHTS_DIR / "RealESRGAN_x2plus.pth"

# CodeFormer
CODEFORMER_WEIGHT = WEIGHTS_DIR / "codeformer.pth"

# Scratch detection
SCRATCH_THRESHOLD = 10  # 顶帽变换阈值，越小检测越灵敏
SCRATCH_KERNEL_SIZE = 15  # 形态学核大小
