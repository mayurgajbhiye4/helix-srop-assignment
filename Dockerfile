FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md ./
COPY app ./app
COPY alembic.ini alembic ./alembic
COPY docs ./docs
COPY .env.example .env

# Install Python dependencies using pip (simpler than uv in Docker)
RUN pip install --no-cache-dir -e .

# Ingest documents into vector store
RUN python -m app.rag.ingest --path docs/

# Expose FastAPI port
EXPOSE 8000

# Run the app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]