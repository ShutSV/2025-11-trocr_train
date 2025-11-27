from fastapi import APIRouter, status, UploadFile, File, HTTPException, Depends
from fastapi.responses import ORJSONResponse
import logging
from src.utils import OCRResponse, ocr_image
from src import get_session_settings
from src.utils import SessionSettingsOCR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/ocr",
    default_response_class=ORJSONResponse,
    tags=["Inference"]
)

@router.get("/")
async def get_ocr_info():
    return {
        "message": "Используйте POST запрос для загрузки изображения",
        "endpoint": "/api/v1/ocr/",
        "method": "POST",
        "parameters": "file: UploadFile"
    }

@router.post(
    path="/",
    status_code=status.HTTP_201_CREATED,
    response_model=OCRResponse,
    name="Загрузка и обработка изображений",
)
async def process_image(
    file: UploadFile = File(...),
    session_settings: SessionSettingsOCR = Depends(get_session_settings)
):
    try:
        return await ocr_image(
            file,
            model_path=session_settings.get_model_path(),
            device=session_settings.get_device()
        )
    except Exception as e:
        logger.error(f"Ошибка обработки изображения {file.filename}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка обработки изображения: {str(e)}"
        )
