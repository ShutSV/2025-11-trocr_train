from pydantic import BaseModel
from typing import Literal, Optional


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


class ConfigOCRTraining(BaseModel):
    model: str
    device: Literal["mps", "cuda", "cpu"] = "cpu"
    epochs: int
    batch_size: int
    learning_rate: float
    max_length: int
    warmup_steps: int
    max_grad_norm: float
    log_interval: int
    num_workers: int
    checkpoint_interval: int
    output_dir: Optional[str] = './output'
    dataset_path: Optional[str] = './datasets'
