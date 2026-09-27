import os
import shutil

from fastapi import APIRouter, File, HTTPException, UploadFile

from app import config, ingest
from app.dependencies import reset_retriever_cache

router = APIRouter()


@router.get("")
def list_documents():
    if not os.path.isdir(config.DATA_PATH):
        return []
    docs = []
    for name in sorted(os.listdir(config.DATA_PATH)):
        if os.path.splitext(name)[1].lower() in config.SUPPORTED_EXTENSIONS:
            path = os.path.join(config.DATA_PATH, name)
            docs.append({"name": name, "size_bytes": os.path.getsize(path)})
    return docs


@router.post("/upload")
def upload_documents(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    unsupported = [f.filename for f in files if os.path.splitext(f.filename or "")[1].lower() not in config.SUPPORTED_EXTENSIONS]
    if unsupported:
        raise HTTPException(status_code=400, detail=f"Unsupported file type(s): {', '.join(unsupported)}")

    os.makedirs(config.DATA_PATH, exist_ok=True)
    saved = []
    for f in files:
        dest = os.path.join(config.DATA_PATH, os.path.basename(f.filename))
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append(f.filename)

    ingest.create_vector_db()
    reset_retriever_cache()
    return {"ingested": saved}
