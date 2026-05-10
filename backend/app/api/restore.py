import json
import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.core.config import OUTPUT_DIR, UPLOAD_DIR
from app.services.pipeline import RestoreParams
from app.services.task_manager import TaskStatus, task_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


@router.post("/restore")
async def restore_photo(
    file: UploadFile = File(...),
    params: str = Form("{}"),
):
    """Submit a restoration task. Returns task_id immediately."""
    ext = Path(file.filename or "image.jpg").suffix or ".jpg"
    uid = uuid.uuid4().hex[:12]
    input_path = UPLOAD_DIR / f"{uid}_input{ext}"
    output_path = OUTPUT_DIR / f"{uid}_output{ext}"

    content = await file.read()
    input_path.write_bytes(content)

    try:
        params_dict = json.loads(params)
        restore_params = RestoreParams(**params_dict)
    except (json.JSONDecodeError, TypeError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid params JSON: {e}") from e
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid params: {e}") from e

    task_id = task_manager.create_task(input_path, output_path, params=restore_params)
    return {"task_id": task_id}


@router.get("/status/{task_id}")
async def get_status(task_id: str):
    """Poll task status and progress."""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    result = {
        "task_id": task.task_id,
        "status": task.status.value,
        "progress": task.progress,
        "step": task.step,
    }
    if task.error:
        result["error"] = task.error
    return result


@router.get("/result/{task_id}")
async def get_result(task_id: str):
    """Retrieve the restored image for a completed task."""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.status != TaskStatus.DONE:
        raise HTTPException(
            status_code=400,
            detail=f"Task status is {task.status.value}, not done",
        )
    if not task.output_path or not task.output_path.exists():
        raise HTTPException(status_code=500, detail="Output file missing")

    ext = task.output_path.suffix
    return FileResponse(
        path=str(task.output_path),
        media_type="image/png",
        filename=f"restored_{task_id}{ext}",
    )


@router.get("/health")
async def health():
    return {"status": "ok"}
