from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import ARTIFACTS_DIR, DATA_DIR, settings
from .data.source import RawTableStore
from .routers import history, insights, score
from .scorer import Scorer


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[startup] Loading 7 raw tables from {DATA_DIR} ...")
    app.state.store = RawTableStore.load(DATA_DIR)
    print(f"[startup] Loading model artifacts from {ARTIFACTS_DIR} ...")
    app.state.scorer = Scorer(ARTIFACTS_DIR, model_version=settings.model_version)
    print("[startup] Ready.")
    yield


app = FastAPI(title="Home Credit Default Risk Scoring API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(score.router)
app.include_router(history.router)
app.include_router(insights.router)


@app.get("/health")
def health():
    return {"status": "ok"}
