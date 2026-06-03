from __future__ import annotations

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path
from dotenv import load_dotenv

_backend_dir = Path(__file__).resolve().parent.parent.parent
_root_dir = _backend_dir.parent
load_dotenv(_root_dir / ".env")
load_dotenv(_backend_dir / ".env", override=True)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

from live_assist.api.routes.live_feedback import router as live_feedback_router
from live_assist.api.routes.rag_documents import router as rag_documents_router
from live_assist.storage.sqlite import init_db

app = FastAPI(title="Live Assist MVP API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(live_feedback_router)
app.include_router(rag_documents_router)


@app.on_event("startup")
async def startup_event() -> None:
    init_db()


@app.get("/health")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
