# Bug fix #9: CMD updated to use uvicorn instead of gunicorn

FROM python:3.11-slim

WORKDIR /app

# Install OS dependencies for LightGBM and uvicorn
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY score.py .

# Non-root user for security
RUN useradd --create-home appuser
USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8080/health').raise_for_status()"

# Bug fix #9: uvicorn replaces gunicorn; workers managed by K8s HPA instead
CMD ["uvicorn", "score:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2", "--log-level", "info"]
