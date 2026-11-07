---
title: German day-ahead power price forecast
sdk: docker
app_port: 7860
pinned: false
short_description: PyTorch quantile forecasts of DE-LU day-ahead prices via FastAPI
---

# German day-ahead power price forecast

Demonstration service on public data: hourly 10%, 50% and 90% quantile forecasts of German (DE-LU)
day-ahead power prices from an ensemble of PyTorch networks (MLP and LSTM), served with FastAPI.
Inputs are fetched from SMARD and public fuel price sources at request time; every forecast is
checked for missing or implausible inputs first.

Endpoints:
- `/health` service and model status
- `/model` members of the ensemble with their training window
- `/forecast?day=YYYY-MM-DD` 24 hourly quantile forecasts for a delivery day
- `/monitoring` latest data checks, drift report and live accuracy of the daily job
- `/docs` interactive API documentation

Code, backtest and method: https://github.com/michaelserikov/de-price-forecast-service

Not investment advice; a research and engineering demonstration.
