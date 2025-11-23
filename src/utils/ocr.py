import logging
import io
from PIL import Image
from kraken import binarization, pageseg
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from ..utils import settings, OCRLine, OCRResponse

# Настройка логирования
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

# # Глобальные переменные - ДОЛЖНЫ БЫТЬ ОБЪЯВЛЕНЫ
# processor = None
# model = None
# device = None


# def initialize_model(model_path=settings.MODEL_PATH, device=settings.DEVICE):
#     """Инициализация модели при запуске сервера"""
#     logger.info(f"Инициализация модели на устройстве {settings.DEVICE}")
#
#     try:
#         # Загрузка модели
#         model_path = model_path
#         processor = TrOCRProcessor.from_pretrained(model_path, use_fast=False)
#         model = VisionEncoderDecoderModel.from_pretrained(model_path)
#         model.to(device)
#         model.eval()
#         logger.info("Модель успешно загружена и инициализирована")
#     except Exception as e:
#         logger.error(f"Ошибка загрузки модели: {e}")
#         raise


async def ocr_image(file, model_path=settings.MODEL_PATH, device=settings.DEVICE):
    """
    Обработка изображения: сегментация и распознавание текста
    """
    # global processor, model
    processor = TrOCRProcessor.from_pretrained(model_path, use_fast=False)
    model = VisionEncoderDecoderModel.from_pretrained(model_path)
    model.to(device)
    model.eval()
    logger.info(f"Модель {model_path} успешно загружена и инициализирована")
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')  # конвертируем в grayscale
    logger.info(f"Начата обработка {file.filename}")
    # Сегментация Kraken
    gray_image = image.convert('L')  # Для сегментации используется grayscale
    binary_image = binarization.nlbin(gray_image)  # Бинаризация
    segmentation_result = pageseg.segment(binary_image)  # Сегментация на строки
    qnty_lines = len(segmentation_result.lines)
    logger.info(f"Сегментация {file.filename} завершена. Найдено строк: {qnty_lines}")
    # Распознавание TrOCR
    ocr_lines = []
    for i, line in enumerate(segmentation_result.lines):
        x1, y1, x2, y2 = line.bbox
        line_image = image.crop((x1, y1, x2, y2))
        pixel_values = processor(images=line_image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)

        with torch.no_grad():
            generated_ids = model.generate(pixel_values)

        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        ocr_line = OCRLine(
            line_number=i + 1,
            bbox=[x1, y1, x2, y2],
            text=generated_text
        )
        ocr_lines.append(ocr_line.dict())

    logger.info(f"Распознавание {file.filename} завершено успешно")

    return OCRResponse(
        model_path=model_path,
        filename=file.filename,
        lines=ocr_lines,
        total_lines=qnty_lines,
        status="success"
    )
