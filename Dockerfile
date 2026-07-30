# ==========================================================================
# BiliRAG Unified Production Dockerfile (Hugging Face & Cloud Run compatible)
# Exposes port 7860 by default and bundles both the python backend app
# and static frontend client into a single self-contained image.
# ==========================================================================

FROM python:3.10-slim

# Prevent python from writing pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/app/data/.cache/huggingface

# Set active working directory
WORKDIR /app

# Install runtime and compiler dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy backend app source code into /app/app
COPY backend/app /app/app

# Copy static frontend assets into /app/frontend
COPY frontend /app/frontend

# Hugging Face Spaces routing expects port 7860 by default
EXPOSE 7860

# Run FastAPI server using uvicorn on port 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
