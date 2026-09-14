"""Pipeline configuration and runtime settings."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Storage & Processing Settings
DATABASE_PATH = os.getenv("DATABASE_PATH", str(BASE_DIR / "medical_pq.db"))
ASSETS_DIR = Path(os.getenv("ASSETS_DIR", str(BASE_DIR / "assets")))
SCHEMA_PATH = BASE_DIR / "schema.sql"

# Invariant Constants from Technical Specification (Updated for 65k Output Window)
RENDER_DPI = int(os.getenv("RENDER_DPI", "200"))
PAGE_CHUNK_SIZE = int(os.getenv("PAGE_CHUNK_SIZE", "10"))
MAX_OUTPUT_TOKENS = 65536


# Ensure assets directory exists
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
