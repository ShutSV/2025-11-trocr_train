from fastapi import APIRouter
from fastapi.responses import ORJSONResponse
from .inference import inference_router
from .training import training_router
from .results import results_router


router = APIRouter(
    prefix="/v1",
    default_response_class=ORJSONResponse
)
router.include_router(router=inference_router)
router.include_router(router=training_router)
router.include_router(router=results_router)
