from __future__ import annotations

import base64
import difflib
import hashlib
import io
import json
import re
import shutil
import threading
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import chromadb
from PIL import Image, UnidentifiedImageError
from pypdf import PdfReader

from .config import (
    ALLOWED_UPLOAD_EXTENSIONS,
    BACKUP_DIR,
    CHAT_DIR,
    CHROMA_DIR,
    CODE_EXTENSIONS,
    COLLECTION_NAME,
    DATA_DIR,
    DB_DIR,
    DEFAULT_SETTINGS,
    EMBED_MODEL,
    IMAGE_EXTENSIONS,
    MANIFEST_PATH,
    OLLAMA_BASE_URL,
    PRIMARY_MODEL,
    SETTINGS_PATH,
    TEXT_EXTENSIONS,
    TMP_DIR,
    VERSION_DIR,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_text(value: str, fallback: str = "") -> str:
    return value.strip() if value and value.strip() else fallback


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("._") or "item"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default


def write_json(path: Path, payload: Any) -> None:
    ensure_directory(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def relative_to_data(path: Path) -> str:
    return path.resolve().relative_to(DATA_DIR.resolve()).as_posix()


@dataclass
class ChunkRecord:
    chunk_id: str
    text: str
    metadata: dict[str, Any]


class OllamaClient:
    def __init__(self, base_url: str = OLLAMA_BASE_URL) -> None:
        self.base_url = base_url.rstrip("/")

    def _post(self, endpoint: str, payload: dict[str, Any], timeout: int = 180) -> dict[str, Any]:
        request = Request(
            url=f"{self.base_url}{endpoint}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Ollama request failed ({exc.code}): {detail}") from exc
        except URLError as exc:
            raise RuntimeError(
                "Ollama is not reachable. Start Ollama locally and pull the required models."
            ) from exc

    def tags(self) -> list[str]:
        request = Request(f"{self.base_url}/api/tags", method="GET")
        try:
            with urlopen(request, timeout=10) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except URLError as exc:
            raise RuntimeError("Ollama is not reachable at http://127.0.0.1:11434.") from exc
        return [item.get("name", "") for item in payload.get("models", []) if item.get("name")]

    def embed(self, texts: list[str]) -> list[list[float]]:
        payload = {"model": EMBED_MODEL, "input": texts}
        response = self._post("/api/embed", payload, timeout=180)
        if "embeddings" in response:
            return response["embeddings"]
        if "embedding" in response:
            return [response["embedding"]]
        raise RuntimeError("Ollama embedding response did not include embeddings.")

    def generate(self, prompt: str, system: str = "", temperature: float = 0.2, json_mode: bool = False) -> str:
        payload = {
            "model": PRIMARY_MODEL,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if json_mode:
            payload["format"] = "json"
        response = self._post("/api/generate", payload, timeout=240)
        return response.get("response", "").strip()


class ContentProcessor:
    def classify(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix in CODE_EXTENSIONS:
            return "code"
        if suffix in IMAGE_EXTENSIONS:
            return "image"
        return "document"

    def extract(self, path: Path, user_description: str = "") -> dict[str, Any]:
        category = self.classify(path)
        if category == "code":
            return self._extract_code(path)
        if category == "image":
            return self._extract_image(path, user_description)
        return self._extract_document(path)

    def _extract_document(self, path: Path) -> dict[str, Any]:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            reader = PdfReader(str(path))
            pages: list[str] = []
            for index, page in enumerate(reader.pages):
                try:
                    text = page.extract_text() or ""
                except Exception:
                    text = ""
                cleaned = safe_text(text)
                if cleaned:
                    pages.append(f"[Page {index + 1}]\n{cleaned}")
            body = "\n\n".join(pages)
        else:
            body = path.read_text(encoding="utf-8", errors="ignore")

        body = safe_text(body, fallback=f"{path.name} could not be parsed into readable text.")
        return {
            "category": "document",
            "raw_text": body,
            "summary_seed": body[:5000],
            "chunks": self._chunk_text(body, chunk_size=1400, overlap=220),
            "extra": {},
        }

    def _extract_image(self, path: Path, user_description: str = "") -> dict[str, Any]:
        try:
            with Image.open(path) as image:
                width, height = image.size
                mode = image.mode
        except (UnidentifiedImageError, OSError):
            width, height, mode = 0, 0, "unknown"

        description = safe_text(user_description, "No user description provided.")
        raw_text = (
            f"Image file: {path.name}\n"
            f"Format: {path.suffix.lower()}\n"
            f"Resolution: {width}x{height}\n"
            f"Mode: {mode}\n"
            f"User description: {description}\n"
            "This image should be retrievable alongside documents and code when relevant."
        )
        return {
            "category": "image",
            "raw_text": raw_text,
            "summary_seed": raw_text,
            "chunks": [raw_text],
            "extra": {"width": width, "height": height, "mode": mode},
        }

    def _extract_code(self, path: Path) -> dict[str, Any]:
        text = path.read_text(encoding="utf-8", errors="ignore")
        text = safe_text(text, fallback=f"{path.name} is empty.")
        analysis = self._extract_code_structure(text, path.suffix.lower())
        chunks = self._chunk_code(text, path.name, analysis)
        return {
            "category": "code",
            "raw_text": text,
            "summary_seed": "\n".join(
                [
                    f"Code file: {path.name}",
                    f"Classes: {', '.join(analysis['classes']) or 'None'}",
                    f"Functions: {', '.join(analysis['functions']) or 'None'}",
                    f"Comments: {analysis['comment_count']}",
                    text[:4000],
                ]
            ),
            "chunks": chunks,
            "extra": analysis,
        }

    def _chunk_text(self, text: str, chunk_size: int = 1400, overlap: int = 220) -> list[str]:
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if len(text) <= chunk_size:
            return [text]

        paragraphs = [piece.strip() for piece in re.split(r"\n\s*\n", text) if piece.strip()]
        chunks: list[str] = []
        current = ""

        for paragraph in paragraphs:
            if len(current) + len(paragraph) + 2 <= chunk_size:
                current = f"{current}\n\n{paragraph}".strip()
                continue

            if current:
                chunks.append(current)
                tail = current[-overlap:] if overlap else ""
                current = f"{tail}\n\n{paragraph}".strip()
                continue

            step = max(chunk_size - overlap, 200)
            pieces = [paragraph[i : i + chunk_size] for i in range(0, len(paragraph), step)]
            chunks.extend(piece.strip() for piece in pieces if piece.strip())

        if current:
            chunks.append(current)

        return chunks

    def _extract_code_structure(self, text: str, suffix: str) -> dict[str, Any]:
        comment_pattern = r"(#.*?$|//.*?$|/\*.*?\*/)"
        comment_count = len(re.findall(comment_pattern, text, flags=re.MULTILINE | re.DOTALL))
        functions: list[str] = []
        classes: list[str] = []

        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("class "):
                name = stripped.split()[1].split("(")[0].split(":")[0]
                classes.append(name)
                continue
            if stripped.startswith(("def ", "async def ")):
                match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", stripped)
                if match:
                    functions.append(match.group(1))
                continue
            if suffix in {".js", ".jsx", ".ts", ".tsx"} and "function " in stripped:
                match = re.search(r"function\s+([A-Za-z_][A-Za-z0-9_]*)", stripped)
                if match:
                    functions.append(match.group(1))
            elif suffix in {".java", ".c", ".cpp", ".cc", ".cxx", ".cs", ".go", ".rs", ".php"}:
                match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", stripped)
                if match and not stripped.startswith(("if", "for", "while", "switch", "catch")):
                    functions.append(match.group(1))

        return {
            "functions": sorted(set(functions))[:80],
            "classes": sorted(set(classes))[:80],
            "comment_count": comment_count,
        }

    def _chunk_code(self, text: str, filename: str, analysis: dict[str, Any]) -> list[str]:
        lines = text.splitlines()
        chunks: list[str] = []
        current: list[str] = []
        current_size = 0

        header = (
            f"File: {filename}\n"
            f"Classes: {', '.join(analysis['classes']) or 'None'}\n"
            f"Functions: {', '.join(analysis['functions']) or 'None'}\n"
        )

        for line in lines:
            boundary = bool(re.match(r"^\s*(class|def|async def|function|export function|export class)\b", line))
            if current and (current_size > 1100 and boundary):
                block = "\n".join(current).strip()
                chunks.append(f"{header}\n```text\n{block}\n```")
                current = []
                current_size = 0

            current.append(line)
            current_size += len(line) + 1

            if current_size > 1600:
                block = "\n".join(current).strip()
                chunks.append(f"{header}\n```text\n{block}\n```")
                overlap = current[-20:] if len(current) > 20 else current[:]
                current = overlap
                current_size = sum(len(item) + 1 for item in current)

        if current:
            block = "\n".join(current).strip()
            chunks.append(f"{header}\n```text\n{block}\n```")

        return chunks


class ManifestStore:
    def __init__(self) -> None:
        ensure_directory(DB_DIR)
        self._data = read_json(
            MANIFEST_PATH,
            {
                "files": {},
                "last_sync_at": None,
                "sync_runs": 0,
                "last_error": "",
            },
        )

    @property
    def files(self) -> dict[str, Any]:
        return self._data.setdefault("files", {})

    def get(self, relative_path: str) -> dict[str, Any] | None:
        return self.files.get(relative_path)

    def upsert(self, relative_path: str, payload: dict[str, Any]) -> None:
        self.files[relative_path] = payload

    def delete(self, relative_path: str) -> dict[str, Any] | None:
        return self.files.pop(relative_path, None)

    def mark_sync(self, error: str = "") -> None:
        self._data["last_sync_at"] = utc_now()
        self._data["sync_runs"] = int(self._data.get("sync_runs", 0)) + 1
        self._data["last_error"] = error

    def save(self) -> None:
        write_json(MANIFEST_PATH, self._data)


class ChatStore:
    def __init__(self) -> None:
        ensure_directory(CHAT_DIR)

    def _path(self, session_id: str) -> Path:
        return CHAT_DIR / f"{normalize_name(session_id)}.json"

    def list_sessions(self) -> list[dict[str, Any]]:
        sessions: list[dict[str, Any]] = []
        for path in sorted(CHAT_DIR.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
            payload = read_json(path, {})
            if not payload:
                continue
            sessions.append(
                {
                    "id": payload.get("id", path.stem),
                    "name": payload.get("name", "Untitled Session"),
                    "mode": payload.get("mode", "garden"),
                    "created_at": payload.get("created_at"),
                    "updated_at": payload.get("updated_at"),
                    "message_count": len(payload.get("messages", [])),
                }
            )
        return sessions

    def load(self, session_id: str) -> dict[str, Any]:
        path = self._path(session_id)
        payload = read_json(path, None)
        if not payload:
            raise FileNotFoundError(f"Chat session {session_id} was not found.")
        return payload

    def save(self, payload: dict[str, Any]) -> dict[str, Any]:
        payload["updated_at"] = utc_now()
        write_json(self._path(payload["id"]), payload)
        return payload

    def create(self, name: str, mode: str = "garden") -> dict[str, Any]:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        session_id = f"chat_{timestamp}"
        payload = {
            "id": session_id,
            "name": safe_text(name, f"{mode.title()} Session"),
            "mode": mode,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "messages": [],
        }
        return self.save(payload)

    def delete(self, session_id: str) -> None:
        path = self._path(session_id)
        if path.exists():
            path.unlink()


class GardenService:
    def __init__(self) -> None:
        for directory in [DATA_DIR, DB_DIR, CHROMA_DIR, CHAT_DIR, BACKUP_DIR, VERSION_DIR, TMP_DIR]:
            ensure_directory(directory)

        self.lock = threading.RLock()
        self.ollama = OllamaClient()
        self.processor = ContentProcessor()
        self.manifest = ManifestStore()
        self.chats = ChatStore()
        self.settings = read_json(SETTINGS_PATH, DEFAULT_SETTINGS.copy())
        self._last_snapshot: dict[str, str] = {}
        self.client = self._create_chroma_client()
        self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)

    def _create_chroma_client(self) -> chromadb.PersistentClient:
        try:
            return chromadb.PersistentClient(path=str(CHROMA_DIR))
        except Exception:
            fallback_dir = DB_DIR / "chroma_runtime"
            ensure_directory(fallback_dir)
            try:
                return chromadb.PersistentClient(path=str(fallback_dir))
            except Exception:
                return chromadb.Client()

    def _is_manifest_record_complete(self, record: dict[str, Any]) -> bool:
        required = ["relative_path", "name", "category", "summary", "chunk_count", "chunk_ids"]
        return all(key in record for key in required)

    def _fallback_record(self, relative_path: str, record: dict[str, Any]) -> dict[str, Any]:
        path = DATA_DIR / relative_path
        category = record.get("category")
        if not category and path.exists():
            category = self.processor.classify(path)
        category = category or "document"

        return {
            "relative_path": relative_path,
            "name": record.get("name") or Path(relative_path).name,
            "sha256": record.get("sha256", ""),
            "size": int(record.get("size", path.stat().st_size if path.exists() else 0)),
            "modified_at": record.get("modified_at", ""),
            "indexed_at": record.get("indexed_at", ""),
            "category": category,
            "tags": record.get("tags", []),
            "summary": record.get("summary", ""),
            "chunk_ids": record.get("chunk_ids", []),
            "chunk_count": int(record.get("chunk_count", 0)),
            "user_description": record.get("user_description", ""),
            "extra": record.get("extra", {}),
            "versions": record.get("versions", []),
        }

    def _get_repaired_record(self, relative_path: str, record: dict[str, Any]) -> dict[str, Any] | None:
        if self._is_manifest_record_complete(record):
            merged = dict(record)
            merged["relative_path"] = relative_path
            return merged

        path = DATA_DIR / relative_path
        if path.exists():
            try:
                self._sync_file(path, relative_path, force=True)
                refreshed = self.manifest.get(relative_path) or {}
                repaired = self._fallback_record(relative_path, refreshed)
                self.manifest.upsert(relative_path, repaired)
                self.manifest.save()
                return repaired
            except Exception:
                fallback = self._fallback_record(relative_path, record)
                self.manifest.upsert(relative_path, fallback)
                self.manifest.save()
                return fallback

        self.manifest.delete(relative_path)
        self.manifest.save()
        return None

    def save_settings(self, updates: dict[str, Any]) -> dict[str, Any]:
        self.settings = {**DEFAULT_SETTINGS, **self.settings, **updates}
        write_json(SETTINGS_PATH, self.settings)
        return self.settings

    def health(self) -> dict[str, Any]:
        tags: list[str] = []
        missing: list[str] = []
        running = True
        message = ""
        try:
            tags = self.ollama.tags()
        except Exception as exc:
            running = False
            message = str(exc)

        if PRIMARY_MODEL not in tags:
            missing.append(PRIMARY_MODEL)
        if EMBED_MODEL not in tags and "nomic-embed-text" not in tags:
            missing.append(EMBED_MODEL)

        stats = self.index_stats()
        return {
            "running": running,
            "message": message,
            "models": {"llm": PRIMARY_MODEL, "embedding": EMBED_MODEL},
            "available_models": tags,
            "missing_models": missing,
            "stats": stats,
            "settings": self.settings,
        }

    def _iter_data_files(self) -> list[Path]:
        files: list[Path] = []
        for path in DATA_DIR.rglob("*"):
            if not path.is_file():
                continue
            if path.name.startswith("."):
                continue
            if path.suffix.lower() not in ALLOWED_UPLOAD_EXTENSIONS:
                continue
            files.append(path)
        return sorted(files)

    def _directory_snapshot(self) -> dict[str, str]:
        snapshot: dict[str, str] = {}
        for path in self._iter_data_files():
            stat = path.stat()
            snapshot[relative_to_data(path)] = f"{stat.st_mtime_ns}:{stat.st_size}"
        return snapshot

    def maybe_sync(self) -> dict[str, Any]:
        with self.lock:
            current = self._directory_snapshot()
            if current == self._last_snapshot:
                return {"changed": False, "indexed": 0, "removed": 0}
            self._last_snapshot = current
        return self.sync_index()

    def sync_index(self, force: bool = False, paths: list[Path] | None = None) -> dict[str, Any]:
        with self.lock:
            files = self._iter_data_files()
            current_paths = {relative_to_data(path): path for path in files}
            if paths:
                path_filter = {relative_to_data(path) for path in paths if path.exists()}
            else:
                path_filter = None

            indexed = 0
            removed = 0
            errors: list[str] = []

            existing_manifest_paths = set(self.manifest.files.keys())
            current_manifest_paths = set(current_paths.keys())
            stale_paths = sorted(existing_manifest_paths - current_manifest_paths)

            for relative_path in stale_paths:
                record = self.manifest.delete(relative_path)
                if record:
                    self._delete_chunks(record.get("chunk_ids", []))
                    removed += 1

            for relative_path, path in current_paths.items():
                if path_filter and relative_path not in path_filter:
                    continue
                try:
                    if self._sync_file(path, relative_path, force=force):
                        indexed += 1
                except Exception as exc:
                    errors.append(f"{relative_path}: {exc}")

            self.manifest.mark_sync(error="\n".join(errors))
            self.manifest.save()
            self._last_snapshot = self._directory_snapshot()

            return {
                "changed": bool(indexed or removed or errors),
                "indexed": indexed,
                "removed": removed,
                "errors": errors,
            }

    def _sync_file(self, path: Path, relative_path: str, force: bool = False) -> bool:
        current_hash = file_sha256(path)
        existing = self.manifest.get(relative_path)
        if existing and existing.get("sha256") == current_hash and not force:
            return False

        if existing and existing.get("category") == "code" and existing.get("sha256") != current_hash:
            self._archive_previous_version(path, existing)

        self._delete_chunks(existing.get("chunk_ids", []) if existing else [])

        user_description = ""
        if existing:
            user_description = existing.get("user_description", "")

        extracted = self.processor.extract(path, user_description=user_description)
        if extracted["category"] == "code" and not self.settings.get("code_aware_indexing", True):
            extracted = {
                "category": "code",
                "raw_text": extracted["raw_text"],
                "summary_seed": extracted["summary_seed"],
                "chunks": self.processor._chunk_text(extracted["raw_text"], chunk_size=1400, overlap=220),
                "extra": extracted.get("extra", {}),
            }
        tags = self._generate_tags(path.name, extracted["summary_seed"])
        chunk_records = self._build_chunk_records(path, relative_path, current_hash, extracted, tags)
        if chunk_records:
            texts = [record.text for record in chunk_records]
            embeddings = self.ollama.embed(texts)
            self.collection.upsert(
                ids=[record.chunk_id for record in chunk_records],
                documents=texts,
                metadatas=[record.metadata for record in chunk_records],
                embeddings=embeddings,
            )

        stat = path.stat()
        payload = {
            "relative_path": relative_path,
            "name": path.name,
            "sha256": current_hash,
            "size": stat.st_size,
            "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            "indexed_at": utc_now(),
            "category": extracted["category"],
            "tags": tags,
            "summary": extracted["summary_seed"][:1200],
            "chunk_ids": [record.chunk_id for record in chunk_records],
            "chunk_count": len(chunk_records),
            "user_description": user_description,
            "extra": extracted.get("extra", {}),
            "versions": existing.get("versions", []) if existing else [],
        }
        self.manifest.upsert(relative_path, payload)
        return True

    def _build_chunk_records(
        self,
        path: Path,
        relative_path: str,
        file_hash: str,
        extracted: dict[str, Any],
        tags: list[str],
    ) -> list[ChunkRecord]:
        category = extracted["category"]
        records: list[ChunkRecord] = []
        for index, text in enumerate(extracted["chunks"]):
            chunk_id = f"{normalize_name(relative_path)}::{file_hash[:12]}::{index}"
            metadata = {
                "relative_path": relative_path,
                "name": path.name,
                "category": category,
                "chunk_index": index,
                "sha256": file_hash,
                "tags_csv": ",".join(tags),
                "is_code": category == "code",
                "is_image": category == "image",
            }
            records.append(ChunkRecord(chunk_id=chunk_id, text=text, metadata=metadata))
        return records

    def _delete_chunks(self, chunk_ids: list[str]) -> None:
        if chunk_ids:
            self.collection.delete(ids=chunk_ids)

    def _archive_previous_version(self, path: Path, existing: dict[str, Any]) -> None:
        ensure_directory(VERSION_DIR)
        version_dir = VERSION_DIR / normalize_name(path.stem)
        ensure_directory(version_dir)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archived_name = f"{stamp}_{existing.get('sha256', 'unknown')[:8]}{path.suffix.lower()}"
        archived_path = version_dir / archived_name
        shutil.copy2(path, archived_path)
        versions = existing.setdefault("versions", [])
        versions.append(
            {
                "archived_path": archived_path.relative_to(DB_DIR).as_posix(),
                "sha256": existing.get("sha256"),
                "archived_at": utc_now(),
            }
        )

    def _generate_tags(self, filename: str, summary_seed: str) -> list[str]:
        if not self.settings.get("auto_tagging", True):
            return []
        prompt = (
            "Return a compact JSON object with a single key named tags.\n"
            "The value must be an array of 3 to 8 short lowercase tags.\n"
            "Focus on topics, domains, entities, and intent.\n\n"
            f"Filename: {filename}\n"
            f"Content:\n{summary_seed[:3000]}"
        )
        try:
            response = self.ollama.generate(prompt=prompt, json_mode=True, temperature=0.1)
            payload = json.loads(response)
            raw_tags = payload.get("tags", [])
            tags = [normalize_name(str(tag).lower()).replace("_", "-") for tag in raw_tags if str(tag).strip()]
            return sorted(set(tags))[:8]
        except Exception:
            seed = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", f"{filename} {summary_seed[:400]}")
            return sorted({item.lower() for item in seed})[:6]

    def update_description(self, relative_path: str, description: str) -> dict[str, Any]:
        with self.lock:
            record = self.manifest.get(relative_path)
            if not record:
                raise FileNotFoundError(f"{relative_path} is not indexed.")
            record["user_description"] = safe_text(description)
            self.manifest.upsert(relative_path, record)
            self.manifest.save()
            path = DATA_DIR / relative_path
            self._sync_file(path, relative_path, force=True)
            self.manifest.save()
            return self.manifest.get(relative_path) or {}

    def list_files(self) -> list[dict[str, Any]]:
        self.maybe_sync()
        files: list[dict[str, Any]] = []
        for relative_path, record in sorted(
            list(self.manifest.files.items()),
            key=lambda item: item[1].get("modified_at", ""),
            reverse=True,
        ):
            payload = self._get_repaired_record(relative_path, dict(record))
            if payload:
                files.append(payload)
        return files

    def index_stats(self) -> dict[str, Any]:
        manifest_files = self.manifest.files
        category_counts = {"document": 0, "image": 0, "code": 0}
        total_size = 0
        total_chunks = 0
        for record in manifest_files.values():
            category = record.get("category", "document")
            category_counts[category] = category_counts.get(category, 0) + 1
            total_size += int(record.get("size", 0))
            total_chunks += int(record.get("chunk_count", 0))

        return {
            "file_count": len(manifest_files),
            "chunk_count": total_chunks,
            "total_size": total_size,
            "category_counts": category_counts,
            "last_sync_at": self.manifest._data.get("last_sync_at"),
            "last_error": self.manifest._data.get("last_error", ""),
        }

    def upload_file(
        self,
        filename: str,
        content: bytes,
        category_hint: str = "",
        description: str = "",
        overwrite: bool = True,
    ) -> dict[str, Any]:
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_UPLOAD_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {suffix}")

        safe_name = normalize_name(Path(filename).stem) + suffix
        target_dir = DATA_DIR / ("code" if category_hint == "code" or suffix in CODE_EXTENSIONS else "docs")
        ensure_directory(target_dir)
        target_path = target_dir / safe_name

        if target_path.exists() and not overwrite:
            stem = target_path.stem
            counter = 1
            while (target_dir / f"{stem}_{counter}{suffix}").exists():
                counter += 1
            target_path = target_dir / f"{stem}_{counter}{suffix}"

        target_path.write_bytes(content)
        relative_path = relative_to_data(target_path)
        existing = self.manifest.get(relative_path) or {}
        existing["user_description"] = safe_text(description)
        self.manifest.upsert(relative_path, existing)
        self.manifest.save()
        sync_result = self.sync_index(paths=[target_path], force=True)
        return {
            "relative_path": relative_path,
            "sync": sync_result,
            "record": self.manifest.get(relative_path),
        }

    def delete_file(self, relative_path: str) -> None:
        with self.lock:
            path = DATA_DIR / relative_path
            record = self.manifest.delete(relative_path)
            if record:
                self._delete_chunks(record.get("chunk_ids", []))
            if path.exists():
                path.unlink()
            self.manifest.save()

    def _embed_query(self, query: str) -> list[float]:
        return self.ollama.embed([query])[0]

    def search(self, query: str, mode: str = "all", top_k: int = 8) -> list[dict[str, Any]]:
        self.maybe_sync()
        if not self.manifest.files:
            return []
        query_embedding = self._embed_query(query)
        where: dict[str, Any] | None = None
        if mode == "code":
            where = {"category": "code"}
        elif mode == "documents":
            where = {"category": "document"}
        elif mode == "images":
            where = {"category": "image"}

        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=max(top_k, 1),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        items: list[dict[str, Any]] = []
        seen: set[str] = set()
        for document, metadata, distance in zip(documents, metadatas, distances):
            relative_path = metadata.get("relative_path", "")
            if relative_path in seen:
                continue
            seen.add(relative_path)
            record = self.manifest.get(relative_path) or {}
            items.append(
                {
                    "relative_path": relative_path,
                    "name": metadata.get("name", relative_path),
                    "category": metadata.get("category", "document"),
                    "score": round(1 - float(distance), 4),
                    "snippet": safe_text(document[:500]),
                    "tags": record.get("tags", []),
                    "summary": record.get("summary", ""),
                }
            )
        return items

    def _context_for_prompt(self, results: list[dict[str, Any]], max_chars: int = 7000) -> str:
        sections: list[str] = []
        used = 0
        for item in results:
            section = (
                f"[{item['category'].upper()}] {item['name']} ({item['relative_path']})\n"
                f"Tags: {', '.join(item.get('tags', [])) or 'none'}\n"
                f"Snippet:\n{item['snippet']}"
            )
            if used + len(section) > max_chars:
                break
            sections.append(section)
            used += len(section)
        return "\n\n".join(sections)

    def related_content(self, query: str, mode: str = "all", top_k: int = 4) -> list[dict[str, Any]]:
        return self.search(query=query, mode=mode, top_k=top_k)

    def create_chat(self, name: str, mode: str = "garden") -> dict[str, Any]:
        return self.chats.create(name=name, mode=mode)

    def list_chats(self) -> list[dict[str, Any]]:
        return self.chats.list_sessions()

    def get_chat(self, session_id: str) -> dict[str, Any]:
        return self.chats.load(session_id)

    def rename_chat(self, session_id: str, name: str) -> dict[str, Any]:
        session = self.chats.load(session_id)
        session["name"] = safe_text(name, session["name"])
        return self.chats.save(session)

    def delete_chat(self, session_id: str) -> None:
        self.chats.delete(session_id)

    def _generate_chat_name(self, user_message: str) -> str:
        prompt = (
            "Create a concise chat title of 3 to 6 words. Return plain text only.\n\n"
            f"Message:\n{user_message[:400]}"
        )
        try:
            title = self.ollama.generate(prompt=prompt, temperature=0.2)
            return safe_text(title.replace('"', "").splitlines()[0], "New Chat")
        except Exception:
            words = re.findall(r"[A-Za-z0-9]+", user_message)[:5]
            return " ".join(words) or "New Chat"

    def chat(self, session_id: str, message: str, mode: str = "garden") -> dict[str, Any]:
        self.maybe_sync()
        session = self.chats.load(session_id)
        session_mode = mode or session.get("mode", "garden")
        session["mode"] = session_mode
        history = session.get("messages", [])
        if not history and session["name"].startswith(("Garden Session", "Code Session", "General Session")):
            session["name"] = self._generate_chat_name(message)

        context_items: list[dict[str, Any]] = []
        context_mode = "all"
        if session_mode == "code":
            context_mode = "code"
            context_items = self.search(message, mode="code", top_k=6)
        elif session_mode == "garden":
            context_mode = "all"
            context_items = self.search(message, mode="all", top_k=6)

        recent_history = history[-10:]
        history_text = "\n".join(
            f"{item['role'].title()}: {item['content']}" for item in recent_history
        )
        context_text = self._context_for_prompt(context_items)

        if session_mode == "general":
            system = (
                "You are a helpful private local AI assistant. Answer using your own knowledge. "
                "Be direct, accurate, and clear."
            )
            prompt = f"Conversation so far:\n{history_text}\n\nUser: {message}\nAssistant:"
        elif session_mode == "code":
            system = (
                "You are a senior software engineer helping the user learn from a local code garden. "
                "Explain logic, surface function/class relationships, answer code questions, and suggest precise improvements."
            )
            prompt = (
                f"Conversation so far:\n{history_text or 'No prior messages.'}\n\n"
                f"Relevant code context:\n{context_text or 'No indexed code context found.'}\n\n"
                f"User question:\n{message}\n\n"
                "Answer with sections when helpful: Explanation, Key Symbols, Improvements, Risks."
            )
        else:
            system = (
                "You are the AI steward of a private lifelong knowledge garden. "
                "Ground answers in retrieved context, connect ideas, and say what is missing when evidence is thin."
            )
            prompt = (
                f"Conversation so far:\n{history_text or 'No prior messages.'}\n\n"
                f"Knowledge garden context:\n{context_text or 'No indexed context found.'}\n\n"
                f"User question:\n{message}\n\n"
                "Respond warmly and practically. Include actionable insights when helpful."
            )

        reply = self.ollama.generate(prompt=prompt, system=system, temperature=0.2)
        related = self.related_content(reply or message, mode=context_mode, top_k=4) if self.settings.get("show_related_content", True) else []
        history.extend(
            [
                {"role": "user", "content": message, "created_at": utc_now()},
                {"role": "assistant", "content": reply, "created_at": utc_now(), "related_content": related},
            ]
        )
        session["messages"] = history
        saved = self.chats.save(session)
        return {
            "session": saved,
            "reply": reply,
            "related_content": related,
            "used_context": context_items,
        }

    def learn_from_code(self, relative_path: str = "", action: str = "summary", question: str = "") -> dict[str, Any]:
        self.maybe_sync()
        if relative_path:
            record = self.manifest.get(relative_path)
            if not record:
                raise FileNotFoundError(f"{relative_path} was not found.")
            results = self.search(record["name"], mode="code", top_k=6)
        else:
            results = self.search(question or "codebase overview", mode="code", top_k=8)

        context = self._context_for_prompt(results, max_chars=9000)
        instructions = {
            "summary": "Summarize the codebase clearly for a developer onboarding into it.",
            "documentation": "Generate practical internal documentation with modules, responsibilities, and usage notes.",
            "refactor": "Suggest concrete refactoring opportunities, risks, and safe next steps.",
            "explain": "Explain the code and answer the user's question precisely.",
        }
        instruction = instructions.get(action, instructions["summary"])
        prompt = (
            f"{instruction}\n\n"
            f"User question: {question or 'Explain the most important code paths.'}\n\n"
            f"Indexed code context:\n{context}"
        )
        content = self.ollama.generate(
            prompt=prompt,
            system="You are a meticulous senior software engineer and teacher.",
            temperature=0.2,
        )
        return {"content": content, "related_code": results}

    def version_history(self, relative_path: str) -> dict[str, Any]:
        record = self.manifest.get(relative_path)
        if not record:
            raise FileNotFoundError(f"{relative_path} was not found.")
        return {"relative_path": relative_path, "versions": record.get("versions", [])}

    def version_diff(self, relative_path: str, archived_relative_path: str) -> dict[str, Any]:
        current_path = DATA_DIR / relative_path
        archived_path = DB_DIR / archived_relative_path
        current_text = current_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        archived_text = archived_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        diff = "\n".join(
            difflib.unified_diff(
                archived_text,
                current_text,
                fromfile=archived_relative_path,
                tofile=relative_path,
                lineterm="",
            )
        )
        return {"diff": diff}

    def export_chat_markdown(self, session_id: str) -> bytes:
        session = self.chats.load(session_id)
        lines = [f"# {session['name']}", "", f"Mode: {session.get('mode', 'garden')}", ""]
        for message in session.get("messages", []):
            lines.append(f"## {message['role'].title()}")
            lines.append(message["content"])
            lines.append("")
        return "\n".join(lines).encode("utf-8")

    def export_garden_zip(self) -> bytes:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for root in [DATA_DIR, CHAT_DIR]:
                for path in root.rglob("*"):
                    if path.is_file():
                        archive.write(path, arcname=path.relative_to(DATA_DIR.parent))
            if MANIFEST_PATH.exists():
                archive.write(MANIFEST_PATH, arcname=MANIFEST_PATH.relative_to(DATA_DIR.parent))
            if SETTINGS_PATH.exists():
                archive.write(SETTINGS_PATH, arcname=SETTINGS_PATH.relative_to(DATA_DIR.parent))
        buffer.seek(0)
        return buffer.read()

    def create_backup(self) -> dict[str, Any]:
        ensure_directory(BACKUP_DIR)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = BACKUP_DIR / f"backup_{stamp}.zip"
        backup_path.write_bytes(self.export_garden_zip())
        return {"backup_file": backup_path.name}

    def restore_backup(self, filename: str, content: bytes) -> dict[str, Any]:
        ensure_directory(TMP_DIR)
        restore_file = TMP_DIR / normalize_name(filename)
        restore_file.write_bytes(content)
        with zipfile.ZipFile(restore_file, "r") as archive:
            for member in archive.infolist():
                target = DB_DIR.parent / member.filename
                resolved = target.resolve()
                if not str(resolved).startswith(str(DB_DIR.parent.resolve())):
                    raise ValueError("Unsafe path in backup archive.")
                archive.extract(member, path=DB_DIR.parent)
        self.manifest = ManifestStore()
        self.settings = read_json(SETTINGS_PATH, DEFAULT_SETTINGS.copy())
        self.sync_index(force=True)
        return {"restored": True}

    def clear_cache_and_rebuild(self) -> dict[str, Any]:
        with self.lock:
            try:
                self.client.delete_collection(COLLECTION_NAME)
            except Exception:
                pass
            self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)
            self.manifest = ManifestStore()
            for record in self.manifest.files.values():
                record["chunk_ids"] = []
                record["chunk_count"] = 0
            self.manifest.save()
            return self.sync_index(force=True)

    def file_preview(self, relative_path: str) -> dict[str, Any]:
        path = DATA_DIR / relative_path
        if not path.exists():
            raise FileNotFoundError(f"{relative_path} was not found.")
        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            mime = f"image/{'jpeg' if suffix == '.jpg' else suffix.lstrip('.')}"
            encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
            return {"type": "image", "content": f"data:{mime};base64,{encoded}"}
        if suffix == ".pdf":
            encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
            return {"type": "pdf", "content": f"data:application/pdf;base64,{encoded}"}
        if suffix in TEXT_EXTENSIONS or suffix in CODE_EXTENSIONS:
            text = path.read_text(encoding="utf-8", errors="ignore")
            return {"type": "text", "content": text[:20000]}
        return {"type": "binary", "content": ""}

    def system_prompt_catalog(self) -> dict[str, str]:
        return {
            "garden": "RAG-grounded knowledge garden mode",
            "code": "Code-aware research and learning mode",
            "general": "General chat using the model's native knowledge",
        }


garden_service = GardenService()
