from fastapi import APIRouter, status, UploadFile, File, HTTPException
from fastapi.responses import ORJSONResponse
import logging
# from src.utils import OCRResponse, ocr_image, settings


# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/datasets",
    default_response_class=ORJSONResponse,
    tags=["Datasets for train models"]
)


# GET эндпоинт для информации
@router.get("/")
async def get_status_info():
    """
    Информация о Datasets for train models # ЗАГЛУШКА #
    """
    return {
        "message": "Эндпойнт перечня датасетов для тренировки моделей по распознаванию изображений",

    }
