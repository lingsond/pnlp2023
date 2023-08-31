"""MD"""
import datasets
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


CACHE_DIR = 'D:/Cache/huggingface'


class XNLIDataModule(LightningDataModule):
    """CD"""
    task_text_field_map = {
        "prompt": ["sentence"],
        "standard": ["premise", "hypothesis"],
    }

    xnli_task_num_labels = {
        "prompt": 3,
        "standard": 3,
    }

    loader_columns = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels",
    ]

    def __init__(self,
        model_name_or_path: str,
        task_name: str = "standard",
        language: str = 'en',
        max_seq_length: int = 128,
        train_batch_size: int = 1,
        eval_batch_size: int = 32,
        **kwargs,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.language = language
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.text_fields = self.task_text_field_map[task_name]
        self.num_labels = self.xnli_task_num_labels[task_name]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def setup(self, stage: str):
        self.dataset = datasets.load_dataset("xnli", self.language, cache_dir=CACHE_DIR)

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                remove_columns=["label"],
            )
            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(type="torch", columns=self.columns)

        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]
        # x = 0

    def prepare_data(self):
        datasets.load_dataset("xnli", self.language, cache_dir=CACHE_DIR)
        AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True, cache_dir=CACHE_DIR)

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset["validation"], batch_size=self.eval_batch_size)
        # if len(self.eval_splits) == 1:
        #     return DataLoader(self.dataset["validation"], batch_size=self.eval_batch_size)
        # if len(self.eval_splits) > 1:
        #     return [
        #         DataLoader(
        #             self.dataset[x], batch_size=self.eval_batch_size
        #         ) for x in self.eval_splits
        #     ]

    def test_dataloader(self):
        return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size)
        # if len(self.eval_splits) == 1:
        #     return DataLoader(
        #         self.dataset["test"], batch_size=self.eval_batch_size)
        # elif len(self.eval_splits) > 1:
        #     return [DataLoader(
        #         self.dataset[x], batch_size=self.eval_batch_size
        #     ) for x in self.eval_splits]

    def convert_to_features(self, example_batch, indices=None):
        # Either encode single sentence or sentence pairs
        if len(self.text_fields) > 1:
            texts_or_text_pairs = list(zip(example_batch[self.text_fields[0]], example_batch[self.text_fields[1]]))
        else:
            texts_or_text_pairs = example_batch[self.text_fields[0]]

        # Tokenize the text/text pairs
        features = self.tokenizer.batch_encode_plus(
            texts_or_text_pairs, max_length=self.max_seq_length, padding='max_length', truncation=True
        )

        # Rename label to labels to make it easier to pass to model forward
        features["labels"] = example_batch["label"]

        return features

    def convert_to_features_concat(self, example_batch, indices=None):
        # Either encode single sentence or sentence pairs
        if len(self.text_fields) > 1:
            texts_or_text_pairs = list(zip(
                example_batch[self.text_fields[0]],
                example_batch[self.text_fields[1]]
            ))
        # else:
        #     texts_or_text_pairs = example_batch[self.text_fields[0]]
        if self.task_name == 'standard':
            pairs = list(zip(
                example_batch[self.text_fields[0]],
                example_batch[self.text_fields[1]]
            ))
            texts = [f'Premise: {x[0]}\nHypothesis: {x[1]}' for x in pairs]
        elif self.task_name == 'prompt':
            pairs = list(zip(example_batch[self.text_fields[0]], example_batch[self.text_fields[1]]))
            texts = [f'Premise: {x[0]}\nHypothesis: {x[1]}' for x in pairs]

        # Tokenize the text/text pairs
        features = self.tokenizer.batch_encode_plus(
            texts, max_length=self.max_seq_length, padding='max_length', truncation=True
        )

        # Rename label to labels to make it easier to pass to model forward
        features["labels"] = example_batch["label"]

        return features


if __name__ == '__main__':
    xnli = XNLIDataModule('xlm-roberta-base')
    xnli.prepare_data()
    xnli.setup('fit')
    for x in iter(xnli.train_dataloader()):
        print(x)
        break