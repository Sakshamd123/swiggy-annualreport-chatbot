import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager

from app.routes import router, init_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup phase
    init_pipeline()
    yield
    # Shutdown phase
    pass

app = FastAPI(
    title="Swiggy Annual Report RAG API",
    description="API for querying the Swiggy Annual Report FY24 using a Hybrid RAG backend.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static assets (CSS, JS) from /static/
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# Include RAG API routes
app.include_router(router, prefix="/api/v1")

# Serve the frontend SPA at GET /
@app.get("/", include_in_schema=False)
async def serve_index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))
