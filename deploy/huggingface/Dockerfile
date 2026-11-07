# Image for a Hugging Face Docker Space: the same service as ../../Dockerfile, run as user 1000 on port 7860
FROM python:3.12-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH \
    DPFS_ROOT=/home/user/app PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /home/user/app

# CPU-only PyTorch keeps the image small; the models are a few hundred kilobytes.
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu

COPY --chown=user pyproject.toml ./
COPY --chown=user src ./src
RUN pip install --no-cache-dir --user .

COPY --chown=user models ./models
COPY --chown=user forecasts ./forecasts
COPY --chown=user results/backtest/metrics.csv ./results/backtest/metrics.csv

EXPOSE 7860
CMD ["uvicorn", "dpfs.service:app", "--host", "0.0.0.0", "--port", "7860"]
