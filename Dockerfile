FROM python:3.12-slim

WORKDIR /app
ENV DPFS_ROOT=/app PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# CPU-only PyTorch keeps the image small; the models are a few hundred kilobytes.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY models ./models

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "dpfs.service:app", "--host", "0.0.0.0", "--port", "8000"]
