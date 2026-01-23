FROM python:3.12-slim

RUN pip install uv

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md

RUN uv sync --frozen --no-install-project

COPY src src/

RUN uv sync --frozen

ENTRYPOINT ["uv", "run", "uvicorn", "src.mlops_loadconsumption.api:app", "--host", "0.0.0.0", "--port", "8000"]
