import os

from fastapi import APIRouter, File, HTTPException, UploadFile

from app import config, db, ingest, paths
from app.dependencies import reset_retriever_cache

router = APIRouter()


def _require_session(session_id: str):
    if db.get_session(session_id) is None:
        raise HTTPException(status_code=404, detail="Session not found")


@router.get("/sessions/{session_id}/documents")
def list_documents(session_id: str):
    _require_session(session_id)
    data_path = paths.session_data_path(session_id)
    if not os.path.isdir(data_path):
        return []
    docs = []
    for name in sorted(os.listdir(data_path)):
        if os.path.splitext(name)[1].lower() in config.SUPPORTED_EXTENSIONS:
            path = os.path.join(data_path, name)
            docs.append({"name": name, "size_bytes": os.path.getsize(path)})
    return docs


@router.post("/sessions/{session_id}/documents/upload")
def upload_documents(session_id: str, files: list[UploadFile] = File(...)):
    _require_session(session_id)
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    unsupported = [f.filename for f in files if os.path.splitext(f.filename or "")[1].lower() not in config.SUPPORTED_EXTENSIONS]
    if unsupported:
        raise HTTPException(status_code=400, detail=f"Unsupported file type(s): {', '.join(unsupported)}")

    data_path = paths.session_data_path(session_id)
    os.makedirs(data_path, exist_ok=True)
    max_bytes = config.MAX_UPLOAD_MB * 1024 * 1024
    saved = []
    current_filename = None
    try:
        for f in files:
            current_filename = f.filename
            dest = os.path.join(data_path, os.path.basename(f.filename))
            size = 0
            with open(dest, "wb") as out:
                while chunk := f.file.read(1024 * 1024):
                    size += len(chunk)
                    if size > max_bytes:
                        raise HTTPException(
                            status_code=400,
                            detail=f"{f.filename} exceeds the {config.MAX_UPLOAD_MB}MB upload limit",
                        )
                    out.write(chunk)
            saved.append(f.filename)
    except HTTPException:
        # Don't leave a half-written/oversized file, or a partial batch
        # that would silently change what gets ingested, behind after a
        # rejected upload.
        for name in {*saved, current_filename}:
            path = os.path.join(data_path, os.path.basename(name))
            if os.path.isfile(path):
                os.remove(path)
        raise

    ingest.create_vector_db(
        data_path=data_path,
        faiss_path=paths.session_faiss_path(session_id),
        bm25_path=paths.session_bm25_path(session_id),
    )
    reset_retriever_cache(session_id)
    return {"ingested": saved}
