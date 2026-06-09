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


@app.get("/health/langfuse")
async def langfuse_health() -> dict[str, str]:
    """
    Check Langfuse connectivity and configuration status.
    
    Returns one of the following statuses:
    - "healthy": Langfuse is configured and connection successful
    - "disabled": Langfuse observability is disabled via configuration
    - "misconfigured": Langfuse credentials not configured
    - "unhealthy": Langfuse authentication failed
    - "error": Langfuse connection error
    
    Validates: Requirements 11.2
    """
    from live_assist.core.config import get_settings
    
    settings = get_settings()
    
    # Check if Langfuse is disabled
    if not settings.langfuse_enabled:
        return {
            "status": "disabled",
            "message": "Langfuse observability is disabled"
        }
    
    # Check if credentials are configured
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        return {
            "status": "misconfigured",
            "message": "Langfuse credentials not configured"
        }
    
    # Attempt to verify connection
    try:
        from langfuse import Langfuse
        
        langfuse = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host
        )
        
        # Perform authentication check
        if langfuse.auth_check():
            return {
                "status": "healthy",
                "message": "Langfuse connection successful",
                "host": settings.langfuse_host
            }
        else:
            return {
                "status": "unhealthy",
                "message": "Langfuse authentication failed"
            }
    except ImportError:
        return {
            "status": "error",
            "message": "Langfuse package not installed"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Langfuse connection error: {str(e)}"
        }


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
