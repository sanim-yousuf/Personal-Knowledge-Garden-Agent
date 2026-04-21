from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from .config import FRONTEND_DIR
from .services import garden_service


class ChatCreateRequest(BaseModel):
    name: str = "New Chat"
    mode: str = "garden"


class ChatRenameRequest(BaseModel):
    name: str


class ChatMessageRequest(BaseModel):
    message: str = Field(min_length=1)
    mode: str = "garden"


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    mode: str = "all"
    top_k: int = 8


class LearnModeRequest(BaseModel):
    relative_path: str = ""
    action: str = "summary"
    question: str = ""


class SettingsUpdateRequest(BaseModel):
    updates: dict[str, Any]


class DescriptionUpdateRequest(BaseModel):
    description: str = ""


app = FastAPI(title="Personal Knowledge Garden", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def frontend_file() -> Path:
    return FRONTEND_DIR / "index.html"


def as_http_error(exc: Exception, status_code: int = 400) -> HTTPException:
    return HTTPException(status_code=status_code, detail=str(exc))


@app.get("/")
def index() -> FileResponse:
    return FileResponse(frontend_file())


@app.get("/api/health")
def health() -> dict[str, Any]:
    return garden_service.health()


@app.post("/api/index/sync")
def sync_index(force: bool = False) -> dict[str, Any]:
    try:
        return garden_service.sync_index(force=force)
    except Exception as exc:
        raise as_http_error(exc)


@app.post("/api/index/rebuild")
def rebuild_index() -> dict[str, Any]:
    try:
        return garden_service.clear_cache_and_rebuild()
    except Exception as exc:
        raise as_http_error(exc)


@app.get("/api/files")
def list_files() -> dict[str, Any]:
    try:
        return {"files": garden_service.list_files()}
    except Exception as exc:
        raise as_http_error(exc)


@app.get("/api/files/preview")
def preview_file(relative_path: str) -> dict[str, Any]:
    try:
        return garden_service.file_preview(relative_path)
    except FileNotFoundError as exc:
        raise as_http_error(exc, status_code=404)
    except Exception as exc:
        raise as_http_error(exc)


@app.post("/api/files/upload")
async def upload_file(
    file: UploadFile = File(...),
    category_hint: str = Form(""),
    description: str = Form(""),
    overwrite: bool = Form(True),
) -> dict[str, Any]:
    try:
        content = await file.read()
        return garden_service.upload_file(
            filename=file.filename or "upload.txt",
            content=content,
            category_hint=category_hint,
            description=description,
            overwrite=overwrite,
        )
    except Exception as exc:
        raise as_http_error(exc)


@app.patch("/api/files/description")
def update_description(relative_path: str, body: DescriptionUpdateRequest) -> dict[str, Any]:
    try:
        return garden_service.update_description(relative_path, body.description)
    except FileNotFoundError as exc:
        raise as_http_error(exc, status_code=404)
    except Exception as exc:
        raise as_http_error(exc)


@app.delete("/api/files")
def delete_file(relative_path: str) -> dict[str, Any]:
    try:
        garden_service.delete_file(relative_path)
        return {"deleted": True}
    except Exception as exc:
        raise as_http_error(exc)


@app.post("/api/search")
def search(body: SearchRequest) -> dict[str, Any]:
    try:
        return {"results": garden_service.search(body.query, mode=body.mode, top_k=body.top_k)}
    except Exception as exc:
        raise as_http_error(exc)


@app.get("/api/chats")
def list_chats() -> dict[str, Any]:
    return {"sessions": garden_service.list_chats()}


@app.post("/api/chats")
def create_chat(body: ChatCreateRequest) -> dict[str, Any]:
    try:
        return garden_service.create_chat(name=body.name, mode=body.mode)
    except Exception as exc:
        raise as_http_error(exc)


@app.get("/api/chats/{session_id}")
def get_chat(session_id: str) -> dict[str, Any]:
    try:
        return garden_service.get_chat(session_id)
    except FileNotFoundError as exc:
        raise as_http_error(exc, status_code=404)


@app.patch("/api/chats/{session_id}")
def rename_chat(session_id: str, body: ChatRenameRequest) -> dict[str, Any]:
    try:
        return garden_service.rename_chat(session_id, body.name)
    except FileNotFoundError as exc:
        raise as_http_error(exc, status_code=404)


@app.delete("/api/chats/{session_id}")
def delete_chat(session_id: str) -> dict[str, Any]:
    try:
        garden_service.delete_chat(session_id)
        return {"deleted": True}
    except Exception as exc:
        raise as_http_error(exc)


@app.post("/api/chats/{session_id}/messages")
def send_message(session_id: str, body: ChatMessageRequest) -> dict[str, Any]:
    try:
        return garden_service.chat(session_id=session_id, message=body.message, mode=body.mode)
    except FileNotFoundError as exc:
        raise as_http_error(exc, status_code=404)
    except Exception as exc:
        raise as_http_error(exc)


@app.get("/api/chats/{session_id}/export")
def export_chat(session_id: str) -> Response:
    try:
        payload = garden_service.export_chat_markdown(session_id)
        return Response(
            content=payload,
            media_type="text/markdown",
            headers={"Content-Disposition": f'attachment; filename="{session_id}.md"'},
        )
    except FileNotFoundError as exc:
        raise as_http_error(exc, status_code=404)
    except Exception as exc:
        raise as_http_error(exc)


@app.post("/api/code/learn")
def learn_from_code(body: LearnModeRequest) -> dict[str, Any]:
    try:
        return garden_service.learn_from_code(
            relative_path=body.relative_path,
            action=body.action,
            question=body.question,
        )
    except FileNotFoundError as exc:
        raise as_http_error(exc, status_code=404)
    except Exception as exc:
        raise as_http_error(exc)


@app.get("/api/code/versions")
def code_versions(relative_path: str) -> dict[str, Any]:
    try:
        return garden_service.version_history(relative_path)
    except FileNotFoundError as exc:
        raise as_http_error(exc, status_code=404)


@app.get("/api/code/diff")
def code_diff(relative_path: str, archived_relative_path: str) -> dict[str, Any]:
    try:
        return garden_service.version_diff(relative_path, archived_relative_path)
    except FileNotFoundError as exc:
        raise as_http_error(exc, status_code=404)


@app.get("/api/export/garden")
def export_garden() -> Response:
    try:
        payload = garden_service.export_garden_zip()
        return Response(
            content=payload,
            media_type="application/zip",
            headers={"Content-Disposition": 'attachment; filename="knowledge-garden.zip"'},
        )
    except Exception as exc:
        raise as_http_error(exc)


@app.post("/api/backup")
def create_backup() -> dict[str, Any]:
    try:
        return garden_service.create_backup()
    except Exception as exc:
        raise as_http_error(exc)


@app.post("/api/restore")
async def restore_backup(file: UploadFile = File(...)) -> dict[str, Any]:
    try:
        return garden_service.restore_backup(file.filename or "backup.zip", await file.read())
    except Exception as exc:
        raise as_http_error(exc)


@app.get("/api/settings")
def get_settings() -> dict[str, Any]:
    return JSONResponse(garden_service.health())


@app.patch("/api/settings")
def update_settings(body: SettingsUpdateRequest) -> dict[str, Any]:
    try:
        settings = garden_service.save_settings(body.updates)
        return {"settings": settings}
    except Exception as exc:
        raise as_http_error(exc)


@app.get("/api/modes")
def modes() -> dict[str, Any]:
    return {"modes": garden_service.system_prompt_catalog()}
