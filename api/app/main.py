from contextlib import asynccontextmanager
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from app import model as ml
from app.schemas import PredictRequest, PredictResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml.load()
    yield


app = FastAPI(title="Predicción Humedad Relativa", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    return ml.predict(request.model_dump())


@app.get("/metrics")
def get_metrics():
    return ml.metrics


@app.get("/correlation")
def get_correlation():
    return ml.correlation


@app.get("/dataset")
def get_dataset(page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=50)):
    return ml.get_dataset_page(page, page_size)
