from fastapi import APIRouter
from fastapi.responses import ORJSONResponse
from .config import router as config_router
from .health import router as health_router
from .start import router as start_router
from .stop import router as stop_router
from .status import router as status_router


router = APIRouter(
    prefix="/training",
    default_response_class=ORJSONResponse
)
router.include_router(router=config_router)
router.include_router(router=health_router)
router.include_router(router=start_router)
router.include_router(router=stop_router)
router.include_router(router=status_router)