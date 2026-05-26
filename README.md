# Predictive HDD Failure
## MLOps проект

## Описание

Проект посвящен анализу телеметрии жестких дисков SMART-параметров и задаче прогнозирования отказов.

Используется датасет Backblaze Hard Drive Stats.

## Задача

Предсказать отказ диска на основе SMART-параметров.

## Данные

- временные ряды по дискам
- SMART метрики
- сильный дисбаланс классов

## EDA

Ноутбук с анализом:
https://colab.research.google.com/drive/1wc4GeiypdB2PPd_MlSkvHtvA4ZMkCUjH?hl=ru



## Drift monitoring

Расчет data drift, target drift и concept drift запускается командой:

```bash
python -m src.monitoring.calculate_drift
```

Результат сохраняется в `metrics/drift_metrics.json`.

- `data_drift`: PSI по window-признакам модели, reference split = train, current split = test.
- `target_drift`: сдвиг доли положительного класса и PSI по target, reference split = train, current split = test.
- `concept_drift`: падение качества модели на test относительно baseline split = val.

Авторы: Даниил Пименов, Ольга Голева
