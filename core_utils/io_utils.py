import os
from datetime import datetime
import json
from typing import Any, Dict, List

def get_timestamp() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def get_versioned_file_path(file_path: str) -> str:
    """Return a version‑incremented path if the file already exists."""
    if not os.path.exists(file_path):
        return file_path
    base, ext = os.path.splitext(file_path)
    version = 1
    new_file_path = f"{base}_version_{version}{ext}"
    while os.path.exists(new_file_path):
        version += 1
        new_file_path = f"{base}_version_{version}{ext}"
    return new_file_path

def load_json(path: str):
    with open(path, 'r', encoding='utf‑8') as f:
        return json.load(f)

def dump_json(path: str, data: Any) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, 'w', encoding='utf‑8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
