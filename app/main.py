from __future__ import annotations

import logging
import os

from fastapi import FastAPI

from app.controllers.kqmlController import router as kqml_router
from app.controllers.searchController import router as search_router

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

app = FastAPI(title="GeoDialogue API", version="0.1.0")
app.include_router(search_router)
app.include_router(kqml_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
