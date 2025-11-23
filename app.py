from fastapi import FastAPI
from src.api import api_router
# from src.utils import initialize_model


app = FastAPI(title="OCR Server", version="1.0.0")
app.include_router(router=api_router)

# @app.on_event("startup")
# async def startup_event():
#     """Инициализация при запуске приложения"""
#     initialize_model()
