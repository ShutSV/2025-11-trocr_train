from pydantic import BaseModel
from typing import Literal


class OCRResponse(BaseModel):
    model_path: str
    filename: str
    lines: list
    total_lines: int
    status: str


class OCRLine(BaseModel):
    line_number: int
    bbox: list
    text: str

class ConfigOCRInference(BaseModel):
    model: str
    device: Literal["mps", "cuda", "cpu"] = "cpu"
    is_custom: bool = False
