import platform
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import default_data_collator
from datasets import Dataset


if platform.system() == 'Windows':
    CACHE_DIR = 'D:/Cache/huggingface'
else:
    CACHE_DIR = '/home/stud/wangsadirdja/.cache/huggingface/'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cuda':
    torch.set_default_dtype(torch.cuda.FloatTensor)
torch.set_default_device(DEVICE)


class XNLIPEFTPreprocessor:
    def __init__(
            self, model_type, max_length, batch_size, train_json, eval_json, test_json
    ):
        self.text_column = 'prompt'
        self.label_column = 'label_text'
        self.classes = ['Yes', 'Maybe', 'No']
        self.max_length = max_length
        self.batch_size = batch_size
        self.dataset_train = Dataset.from_json(train_json)
        self.dataset_eval = Dataset.from_json(eval_json)
        self.dataset_test = Dataset.from_json(test_json)
        self.model_name = f'bigscience/bloomz-{model_type}'
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, cache_dir=CACHE_DIR
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.target_max_length = max(
            [len(self.tokenizer(class_label)["input_ids"])
             for class_label in self.classes]
        )

    def preprocess_function_trainset(self, examples):
        """Preprocessing the input: tokenizing, concatenating and padding"""
        tokenizer = self.tokenizer
        batch_size = len(examples[self.text_column])
        inputs = [x for x in examples[self.text_column]]
        targets = [str(x) for x in examples[self.label_column]]
        model_inputs = tokenizer(inputs)
        labels = tokenizer(targets)
        # First, we want to concatenate the text and the label
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i] + [tokenizer.pad_token_id]
            label_input_ids = labels["input_ids"][i] + [tokenizer.pad_token_id]
            model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
            labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
            model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
        # Then, we do left padding (for Bloom model, we do left padding)
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i]
            model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
                self.max_length - len(sample_input_ids)
            ) + sample_input_ids
            model_inputs["attention_mask"][i] = [0] * (
                    self.max_length - len(sample_input_ids)
            ) + model_inputs["attention_mask"][i]
            labels["input_ids"][i] = [-100] * (self.max_length - len(sample_input_ids)) + label_input_ids
            model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:self.max_length])
            model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:self.max_length])
            labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:self.max_length])
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def preprocess_function_testset(self, examples):
        """Preprocessing the input: tokenizing and padding.
        For the test set, we don't need the label"""
        tokenizer = self.tokenizer
        batch_size = len(examples[self.text_column])
        inputs = [x for x in examples[self.text_column]]
        model_inputs = tokenizer(inputs)
        # print(model_inputs)
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
                self.max_length - len(sample_input_ids)
            ) + sample_input_ids
            model_inputs["attention_mask"][i] = [0] * (
                    self.max_length - len(sample_input_ids)
            ) + model_inputs["attention_mask"][i]
            model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:self.max_length])
            model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:self.max_length])
        return model_inputs

    def get_processed_dataset(self):
        train_dataset = self.dataset_train.map(
            self.preprocess_function_trainset,
            batched=True,
            num_proc=1,
            remove_columns=self.dataset_train.column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )
        eval_dataset = self.dataset_eval.map(
            self.preprocess_function_trainset,
            batched=True,
            num_proc=1,
            remove_columns=self.dataset_eval.column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )
        test_dataset = self.dataset_test.map(
            self.preprocess_function_testset,
            batched=True,
            num_proc=1,
            remove_columns=self.dataset_test.column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )

        if DEVICE == 'cpu':
            memory_pin = True
        else:
            memory_pin = False

        train_dataloader = DataLoader(
            train_dataset, shuffle=False, collate_fn=default_data_collator,
            generator=torch.Generator(device=DEVICE),
            batch_size=self.batch_size, pin_memory=memory_pin
        )
        eval_dataloader = DataLoader(
            eval_dataset, shuffle=False, collate_fn=default_data_collator,
            generator=torch.Generator(device=DEVICE),
            batch_size=self.batch_size, pin_memory=memory_pin
        )
        test_dataloader = DataLoader(
            test_dataset, shuffle=False, collate_fn=default_data_collator,
            generator=torch.Generator(device=DEVICE),
            batch_size=self.batch_size, pin_memory=memory_pin
        )

        return train_dataloader, eval_dataloader, test_dataloader


if __name__ == '__main__':
    # HYPERPARAMS
    model_params = '560m'
    max_length = 128
    batch_size = 8
    ftrain = 'xnli_train_en_en_nli01_120.jsonl'
    feval = 'xnli_validation_en_en_nli01_120.jsonl'
    ftest = 'xnli_test_en_en_nli01_120.jsonl'

    xnli = XNLIPEFTPreprocessor(
        model_type=model_params, max_length=max_length, batch_size=batch_size,
        train_json=ftrain, eval_json=feval, test_json=ftest
    )

    xnli.get_processed_dataset()
