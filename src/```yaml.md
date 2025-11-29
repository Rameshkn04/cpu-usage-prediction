```yaml
# ...existing code...
train_model:
  cmd: python [train.py](http://_vscodecontentref_/2) --data [data.csv](http://_vscodecontentref_/3) --out models/rf_model.joblib --metrics metrics/metrics.json --model rf
  deps:
    - [train.py](http://_vscodecontentref_/4)
    - [data.csv](http://_vscodecontentref_/5)
  outs:
    - models/rf_model.joblib
  metrics:
    - metrics/metrics.json
# ...existing code...
```