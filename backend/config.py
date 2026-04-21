from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
FRONTEND_DIR = ROOT_DIR / "frontend"
DATA_DIR = ROOT_DIR / "data"
DB_DIR = ROOT_DIR / "db"
CHROMA_DIR = DB_DIR / "chroma"
CHAT_DIR = DB_DIR / "chats"
BACKUP_DIR = DB_DIR / "backups"
VERSION_DIR = DB_DIR / "versions"
TMP_DIR = DB_DIR / "tmp"

OLLAMA_BASE_URL = "http://127.0.0.1:11434"
PRIMARY_MODEL = "gemma3:4b"
EMBED_MODEL = "nomic-embed-text:latest"
COLLECTION_NAME = "knowledge_garden"
MANIFEST_PATH = DB_DIR / "manifest.json"
SETTINGS_PATH = DB_DIR / "settings.json"

TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".rst",
    ".pdf",
    ".json",
    ".yaml",
    ".yml",
    ".csv",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CODE_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".c",
    ".cpp",
    ".cc",
    ".cxx",
    ".h",
    ".hpp",
    ".cs",
    ".go",
    ".rs",
    ".php",
    ".rb",
    ".swift",
    ".kt",
    ".kts",
    ".scala",
    ".sh",
    ".bash",
    ".ps1",
    ".sql",
    ".html",
    ".css",
    ".scss",
    ".sass",
    ".xml",
}

ALLOWED_UPLOAD_EXTENSIONS = TEXT_EXTENSIONS | IMAGE_EXTENSIONS | CODE_EXTENSIONS

DEFAULT_SETTINGS = {
    "theme": "dark",
    "show_related_content": True,
    "auto_tagging": True,
    "code_aware_indexing": True,
    "general_chat_default": False,
}
