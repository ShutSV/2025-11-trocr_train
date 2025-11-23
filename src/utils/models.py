from pydantic import BaseModel


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
