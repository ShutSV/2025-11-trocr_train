from fastapi import APIRouter, status, UploadFile, File, HTTPException
from fastapi.responses import ORJSONResponse
import logging
# from src.utils import OCRResponse, ocr_image, settings


# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/models",
    default_response_class=ORJSONResponse,
    tags=["Trained models"]
)


# GET эндпоинт для информации
@router.get("/")
async def get_status_info():
    """
    Информация о Trained models эндпоинте # ЗАГЛУШКА #
    """
    return {
        "message": "Эндпойнт для перечня моделей по распознаванию изображений",

    }
