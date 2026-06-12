# Legacy Code

Эта папка хранит ранние версии training/data/split кода, которые больше не
используются в production DVC pipeline.

Актуальный production path находится в:

- `src/data/make_dataset.py`
- `src/models/train_baseline.py`
- `src/models/evaluate_baseline.py`
- `src/monitoring/calculate_drift.py`
- `src/api/main.py`

Legacy-файлы сохранены для истории проекта и сравнения подходов. Не импортируйте
их из production-кода без отдельного решения о возврате старой логики.
