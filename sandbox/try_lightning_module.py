from datetime import datetime
from typing import Optional

import datasets
import evaluate
import torch
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from try_datamodule import XNLIDataModule


CACHE_DIR = 'D:/Cache/huggingface'


class XNLITransformer(LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        task_name: str,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        train_batch_size: int = 1,
        eval_batch_size: int = 32,
        eval_splits: Optional[list] = None,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()
        if CACHE_DIR == '':
            self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)
        else:
            self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels, cache_dir=CACHE_DIR)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config, cache_dir=CACHE_DIR)
        # self.metric = datasets.load_metric(
        self.metric = evaluate.load(
            "xnli", experiment_id=datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        )

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]

        if self.hparams.num_labels > 1:
            preds = torch.argmax(logits, axis=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()

        labels = batch["labels"]
        self.log("val_loss", val_loss, prog_bar=True)
        self.log_dict(self.metric.compute(predictions=preds, references=labels), prog_bar=True)

        return {"loss": val_loss, "preds": preds, "labels": labels}

    # def on_validation_epoch_end(self, outputs):
    #     if self.hparams.task_name == "standard":
    #         for i, output in enumerate(outputs):
    #             # matched or mismatched
    #             split = self.hparams.eval_splits[i].split("_")[-1]
    #             preds = torch.cat([x["preds"] for x in output]).detach().cpu().numpy()
    #             labels = torch.cat([x["labels"] for x in output]).detach().cpu().numpy()
    #             loss = torch.stack([x["loss"] for x in output]).mean()
    #             self.log(f"val_loss_{split}", loss, prog_bar=True)
    #             split_metrics = {
    #                 f"{k}_{split}": v for k, v in self.metric.compute(predictions=preds, references=labels).items()
    #             }
    #             self.log_dict(split_metrics, prog_bar=True)
    #         return loss
    #
    #     preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
    #     labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
    #     loss = torch.stack([x["loss"] for x in outputs]).mean()
    #     self.log("val_loss", loss, prog_bar=True)
    #     self.log_dict(self.metric.compute(predictions=preds, references=labels), prog_bar=True)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]


if __name__ == '__main__':
    seed_everything(42)
    dm = XNLIDataModule(
        model_name_or_path="xlm-roberta-base",
    )
    dm.setup("fit")
    model = XNLITransformer(
        model_name_or_path="xlm-roberta-base",
        num_labels=dm.num_labels,
        eval_splits=dm.eval_splits,
        task_name=dm.task_name,
    )

    trainer = Trainer(
        max_epochs=1,
        accelerator="gpu" if torch.cuda.is_available() else "auto",
        # devices="auto" if torch.cuda.is_available() else None,  # limiting got iPython runs
        devices=1
    )
    trainer.fit(model, datamodule=dm)
    trainer.validate(model, dm)
    trainer.test(model, dm)
