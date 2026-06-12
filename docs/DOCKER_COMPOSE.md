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

## Что используется контейнером

- `models/` копируется в Docker image и используется API для загрузки модели.
- `./mlruns:/app/mlruns` используется для локального хранилища MLflow, если оно нужно при отладке.
- `PYTHONUNBUFFERED=1`
- `PROMETHEUS_URL` и `GRAFANA_URL` можно задать для корректных ссылок в веб UI.
- при необходимости логов можно добавить: `./logs:/app/logs` в `docker-compose.yml`.

## Проверка совместимости

Compose собирается из существующего `Dockerfile` и использует текущую структуру:
- `src/api/` (`uvicorn src.api.main:app` из `Dockerfile`)
- `models/` для модельных артефактов
- `mlruns/` для локального трекинга MLflow
