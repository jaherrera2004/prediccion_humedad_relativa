# Modelo — Árbol de Decisión

## Variable objetivo

`Humidity` (continua, 0.0–1.0) discretizada en 3 categorías:

| Clase | Rango | Distribución |
|---|---|---|
| Alta | ≥ 0.70 | 64% |
| Media | 0.40 – 0.70 | 28% |
| Baja | < 0.40 | 8% |

Desbalance de clases manejado con `class_weight="balanced"`.

---

## Features de entrada

| Feature | Descripción |
|---|---|
| `temperature` | Temperatura del aire (°C) |
| `apparent_temperature` | Temperatura aparente (°C) |
| `wind_speed` | Velocidad del viento (km/h) |
| `wind_bearing` | Dirección del viento (°) |
| `visibility` | Visibilidad (km) |
| `pressure` | Presión atmosférica (mbar) |
| `mes` | Mes del registro (generado por el backend) |
| `hora` | Hora del registro (generado por el backend) |
| `dia_semana` | Día de la semana (generado por el backend) |

---

## Importancia de variables

| Feature | Importancia |
|---|---|
| apparent_temperature | 29.1% |
| hora | 13.2% |
| visibility | 13.0% |
| temperature | 12.3% |
| pressure | 9.5% |
| wind_speed | 7.1% |
| mes | 6.4% |
| wind_bearing | 6.1% |
| dia_semana | 3.3% |

---

## Métricas

| Métrica | Valor |
|---|---|
| Accuracy | 82.5% |
| F1 Macro | 78.3% |
| CV F1 Macro (5-fold) | 71.0% ± 1.6% |

| Clase | Precision | Recall | F1 |
|---|---|---|---|
| Alta | 89.2% | 88.7% | 88.9% |
| Baja | 76.2% | 77.4% | 76.8% |
| Media | 68.8% | 69.2% | 69.0% |

---

## Hiperparámetros (GridSearchCV)

```
criterion:          entropy
max_depth:          None
min_samples_leaf:   1
min_samples_split:  2
```

---

## API

| Endpoint | Método | Descripción |
|---|---|---|
| `/predict` | POST | Recibe features, retorna clase + probabilidades + regla |
| `/metrics` | GET | Retorna métricas del modelo |
| `/docs` | GET | Swagger UI (FastAPI) |
