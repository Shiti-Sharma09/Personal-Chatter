import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import db
from app.routers import chat, documents, sessions


@asynccontextmanager
async def lifespan(_app: FastAPI):
    db.init_db()
    yield


app = FastAPI(title="Personal Chatter API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    # Vite's default dev server port. In production the frontend is
    # served from this same origin (see the StaticFiles mount below), so
    # CORS only matters for local development.
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(sessions.router, prefix="/api/sessions", tags=["sessions"])
app.include_router(chat.router, prefix="/api", tags=["chat"])
app.include_router(documents.router, prefix="/api/documents", tags=["documents"])


@app.get("/api/health")
def health():
    return {"status": "ok"}


# In production, `npm run build` in frontend/ produces frontend/dist;
# mounting it here lets a single `uvicorn app.main:app` serve both the
# API and the UI. In dev, run the Vite dev server separately instead
# (it proxies /api to this backend -- see frontend/vite.config.ts) and
# this mount is simply skipped since dist/ won't exist yet.
_frontend_dist = os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "dist")
if os.path.isdir(_frontend_dist):
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=_frontend_dist, html=True), name="frontend")
