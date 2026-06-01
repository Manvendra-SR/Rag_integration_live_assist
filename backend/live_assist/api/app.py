from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from live_assist.api.routes.live_feedback import router as live_feedback_router
from live_assist.api.routes.rag_pipeline import router as rag_pipeline_router
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
app.include_router(rag_pipeline_router)


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
