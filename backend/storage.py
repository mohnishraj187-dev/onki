from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass
class ScanRecord:
    scan_id: str
    original_filename: str
    nifti_path: Path
    mask_path: Path
    brain_glb: Path
    tumor_glb: Path
    dimensions: dict
    segmentation: str
    tumor_volume_cm3: float


SCANS: Dict[str, ScanRecord] = {}


def add_scan(record: ScanRecord) -> None:
    SCANS[record.scan_id] = record


def get_scan(scan_id: str) -> Optional[ScanRecord]:
    return SCANS.get(scan_id)
