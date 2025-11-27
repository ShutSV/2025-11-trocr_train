import logging
import io
from PIL import Image
from kraken import binarization, pageseg
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from . import OCRLine, OCRResponse


logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

# Кэш для моделей (опционально, для производительности)
_model_cache = {}


def get_cached_model(model_path: str, device: str):
    """Получить модель из кэша или загрузить новую"""
    cache_key = f"{model_path}_{device}"

    if cache_key not in _model_cache:
        logger.info(f"Загрузка модели {model_path} на устройство {device}")
        processor = TrOCRProcessor.from_pretrained(model_path, use_fast=False)
        model = VisionEncoderDecoderModel.from_pretrained(model_path)
        model.to(device)
        model.eval()
        _model_cache[cache_key] = (processor, model)
        logger.info(f"Модель {model_path} загружена в кэш")

    return _model_cache[cache_key]


async def ocr_image(file, model_path: str, device: str):
    """
    Обработка изображения: сегментация и распознавание текста
    """
    # Используем кэшированную модель или загружаем новую
    processor, model = get_cached_model(model_path, device)

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    logger.info(f"Начата обработка {file.filename}")

    # Сегментация Kraken
    gray_image = image.convert('L')
    binary_image = binarization.nlbin(gray_image)
    segmentation_result = pageseg.segment(binary_image)
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
