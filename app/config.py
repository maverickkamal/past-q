"""Pipeline configuration and runtime settings."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# Storage & Processing Settings
DATABASE_PATH = os.getenv("DATABASE_PATH", str(BASE_DIR / "medical_pq.db"))
ASSETS_DIR = Path(os.getenv("ASSETS_DIR", str(BASE_DIR / "assets")))
SCHEMA_PATH = BASE_DIR / "schema.sql"
CURRICULUM_TAXONOMY_PATH = (
    BASE_DIR / "app" / "curriculum_taxonomy.json"
    if (BASE_DIR / "app" / "curriculum_taxonomy.json").exists()
    else BASE_DIR / "src" / "curriculum_taxonomy.json"
)

# Invariant Constants from Technical Specification (Updated for 65k Output Window)
RENDER_DPI = int(os.getenv("RENDER_DPI", "200"))
PAGE_CHUNK_SIZE = int(os.getenv("PAGE_CHUNK_SIZE", "10"))
MAX_OUTPUT_TOKENS = 65536


# TypeSafe AI Settings
TYPESAFE_API_KEY = os.getenv("TYPESAFE_API_KEY")
TYPESAFE_MODEL = os.getenv("TYPESAFE_MODEL", "jev-latest")

# Ensure assets directory exists
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
