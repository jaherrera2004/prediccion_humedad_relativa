from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    temperature:          float = Field(..., ge=-50,  le=60)
    apparent_temperature: float = Field(..., ge=-50,  le=60)
    wind_speed:           float = Field(..., ge=0,    le=200)
    wind_bearing:         float = Field(..., ge=0,    le=360)
    visibility:           float = Field(..., ge=0,    le=100)
    pressure:             float = Field(..., ge=800,  le=1100)


class PredictResponse(BaseModel):
    clase:          str
    probabilidades: dict[str, float]
    regla:          str
