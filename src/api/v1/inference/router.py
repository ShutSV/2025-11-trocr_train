from fastapi import APIRouter
from fastapi.responses import ORJSONResponse
from .config import router as config_router
from .ocr import router as ocr_router


router = APIRouter(
    prefix="/inference",
    default_response_class=ORJSONResponse
)
router.include_router(router=config_router)
router.include_router(router=ocr_router)
