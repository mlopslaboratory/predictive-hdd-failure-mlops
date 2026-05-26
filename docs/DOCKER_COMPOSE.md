# Docker Compose локальный запуск FastAPI

Для локального запуска API используется `docker-compose.yml` (Compose v2).

## Запустить

```bash
docker compose up --build
```

## Остановить

```bash
docker compose down
```

## Смотреть логи

```bash
docker compose logs -f
```

## Открыть Swagger UI

`http://localhost:8000/docs`

## Что смонтировано в контейнер

- `./artifacts:/app/artifacts`
- `./mlruns:/app/mlruns` (локальное хранилище MLflow)
- `PYTHONUNBUFFERED=1`
- при необходимости логов можно добавить: `./logs:/app/logs` в `docker-compose.yml`

## Проверка совместимости

Compose собирается из существующего `Dockerfile` и использует текущую структуру:
- `src/api/` (`uvicorn src.api.main:app` из `Dockerfile`)
- `artifacts/` для модельных артефактов
- `mlruns/` для локального трекинга MLflow

