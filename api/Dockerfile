FROM python:3.11-slim

WORKDIR /project

COPY api/requirements.txt api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt

COPY api/app api/app
COPY models  models
COPY data/weatherHistory_clean.csv data/weatherHistory_clean.csv

WORKDIR /project/api
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
