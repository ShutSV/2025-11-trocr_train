from fastapi import APIRouter
from fastapi.responses import ORJSONResponse
from .models import models_router
from .datasets import datasets_router


router = APIRouter(
    prefix="/results",
    default_response_class=ORJSONResponse
)
router.include_router(router=models_router)
router.include_router(router=datasets_router)
