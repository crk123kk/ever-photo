import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.core.config import OUTPUT_DIR, UPLOAD_DIR
from app.services.pipeline import pipeline

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


@router.post("/restore")
async def restore_photo(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix or ".jpg"
    uid = uuid.uuid4().hex[:12]
    input_path = UPLOAD_DIR / f"{uid}_input{ext}"
    output_path = OUTPUT_DIR / f"{uid}_output{ext}"

    content = await file.read()
    input_path.write_bytes(content)

    try:
        pipeline.restore(str(input_path), str(output_path))
    except Exception as e:
        logger.exception("Restoration failed")
        raise HTTPException(status_code=500, detail=str(e)) from e

    return FileResponse(
        path=str(output_path),
        media_type="image/png",
        filename=f"restored_{uid}{ext}",
    )


@router.get("/health")
async def health():
    return {"status": "ok"}
