import datasets
import evaluate
from datetime import datetime



metrics_old = datasets.load_metric(
            "xnli", experiment_id=datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        )
print(metrics_old)

metrics = evaluate.load('xnli')
print(metrics)
