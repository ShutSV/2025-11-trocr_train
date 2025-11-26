from fastapi import APIRouter, status, UploadFile, File, HTTPException
from fastapi.responses import ORJSONResponse
import logging
# from src.utils import OCRResponse, ocr_image, settings


# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/status",
    default_response_class=ORJSONResponse,
    tags=["OCR"]
)


# GET эндпоинт для информации
@router.get("/")
async def get_status_info():
    """
    Информация о OCR эндпоинте
    """
    return {
        "message": "Используйте POST запрос для загрузки изображения",
        "endpoint": "/api/v1/ocr/",
        "method": "POST",
        "parameters": "file: UploadFile"
    }
