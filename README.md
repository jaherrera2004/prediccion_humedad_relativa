# Predicción de Humedad Relativa — Árbol de Decisión

Modelo de clasificación que predice el nivel de humedad relativa (Baja / Media / Alta) a partir de variables meteorológicas. Dataset: Weather in Szeged 2006–2016.

---

## Stack

| Capa | Tecnología |
|---|---|
| Machine Learning | scikit-learn — DecisionTreeClassifier |
| Backend | FastAPI + Uvicorn |
| Frontend | HTML + CSS + JS (nginx) |
| Infraestructura | Docker + Docker Compose |

---

## Inicio rápido

```bash
docker compose up --build
```

- Dashboard: http://localhost
- API docs: http://localhost:8000/docs

---

## Ejecución manual (desarrollo)

```bash
# 1. Limpiar datos
python scripts/clean_data.py

# 2. Entrenar modelo y exportar artefactos
python scripts/train.py

# 3. Levantar API
cd api && uvicorn app.main:app --reload --port 8000

# 4. Abrir frontend/index.html en el navegador
```

---

## Estructura

```
mineria_proyecto/
├── data/
│   ├── weatherHistory.csv          # Dataset original
│   └── weatherHistory_clean.csv   # Dataset limpio
├── models/
│   ├── model.pkl                   # Modelo entrenado
│   ├── features.json               # Orden de features
│   └── metrics.json                # Métricas de evaluación
├── scripts/
│   ├── clean_data.py               # Limpieza del dataset
│   └── train.py                    # Entrenamiento y evaluación
├── api/
│   ├── app/
│   │   ├── main.py                 # Rutas FastAPI
│   │   ├── model.py                # Carga y predicción
│   │   └── schemas.py              # Schemas Pydantic
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── index.html
│   ├── style.css
│   ├── app.js
│   └── Dockerfile
├── docs/
│   └── modelo.md                   # Detalles del modelo
└── docker-compose.yml
```
