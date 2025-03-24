from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    local_data_directory: Path
    source_url: str


@dataclass(frozen=True)
class TurboProcessorConfig:
    input_dir: Path
    output_dir: Path
    T_d: int
    T_window: int
    eps: int