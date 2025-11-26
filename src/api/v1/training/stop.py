from fastapi import APIRouter, status, UploadFile, File, HTTPException
from fastapi.responses import ORJSONResponse
import logging
from src.utils import OCRResponse, ocr_image
from src.utils import settings_train as settings


# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/stop",
    default_response_class=ORJSONResponse,
    tags=["Training models"],
)


# GET эндпоинт для информации
@router.get("/")
async def get_ocr_info():
    """
    Информация об эндпоинте для остановки обучения модели # ЗАГЛУШКА #
    """
    return {
        "message": "Используйте POST запрос для остановки обучения",
        "endpoint": "/api/v1/training/train",
        "method": "POST",
        "parameters": "file: UploadFile"
    }


@router.post(
    path="/",
    status_code=status.HTTP_201_CREATED,
    response_model=OCRResponse,
    name="Остановка обучения модели",
)

async def process_image(file: UploadFile = File(...)):
    """
    Остановка обучения модели для распознавания текста # ЗАГЛУШКА #
    """
    try:
        return await ocr_image(file, model_path=settings.MODEL_PATH, device=settings.DEVICE)
    except Exception as e:
        logger.error(f"Ошибка обработки изображения {file.filename}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Ошибка обработки изображения: {str(e)}")
