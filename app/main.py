from __future__ import annotations

from fastapi import FastAPI

from app.controllers.kqmlController import router as kqml_router
from app.controllers.searchController import router as search_router

app = FastAPI(title="GeoDialogue API", version="0.1.0")
app.include_router(search_router)
app.include_router(kqml_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
