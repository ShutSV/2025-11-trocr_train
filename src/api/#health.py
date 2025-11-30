import datetime
from typing import Dict
from fastapi import APIRouter, Request, status
from fastapi.responses import ORJSONResponse


router = APIRouter(
    prefix="/",
    default_response_class=ORJSONResponse,
    tags=["Health"]
)


@router.get(
    path="/",
    status_code=status.HTTP_200_OK,
    response_model=Dict,
    name="Health check with endpoints tree",
)
async def health_check(request: Request):
    app = request.app

    endpoints = []
    for route in app.routes:
        if hasattr(route, "methods") and hasattr(route, "path"):
            endpoints.append({
                "path": route.path,
                "methods": list(route.methods) if hasattr(route, "methods") else [],
                "name": getattr(route, "name", None)
            })

    return {
        "status": "healthy",
        "timestamp": datetime.datetime.now().isoformat(),
        "endpoints_tree": endpoints,
        "total": len(endpoints)
    }
