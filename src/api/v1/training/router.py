from fastapi import APIRouter
from fastapi.responses import ORJSONResponse
from .config import router as config_router
from .pause import router as pause_router
from .start import router as start_router
from .stop import router as stop_router
from .resume import router as resume_router
from .status import router as status_router
from .history import router as history_router


router = APIRouter(
    prefix="/training",
    default_response_class=ORJSONResponse
)
router.include_router(router=config_router)
router.include_router(router=pause_router)
router.include_router(router=start_router)
router.include_router(router=stop_router)
router.include_router(router=status_router)
router.include_router(router=history_router)
router.include_router(router=resume_router)
