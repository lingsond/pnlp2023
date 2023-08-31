"""Module Documentation"""
from statistics import pstdev, mean
from typing import Optional
import json
import os

from datasets import get_dataset_split_names, get_dataset_config_names, \
    load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm


CACHE_DIR = 'D:/Cache/huggingface'


class NLIAnalyzer:
    """Class Documentation"""
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.dataset = {}
        self.features = [['premise', 'hypothesis'], 'label']
        self.languages = self.get_languages()
        self.split_names = get_dataset_split_names(self.dataset_name, 'all_languages')

    def get_languages(self):
        """Method"""
        config_names = get_dataset_config_names(self.dataset_name)
        languages = [x for x in config_names if len(x) <= 3]
        return languages

    def count_max_token(self, lang: str, tokenizer: AutoTokenizer) -> dict:
        """Method"""
        tokens_length = {}
        for split in self.split_names:
            tokens = []
            # feats = [(x['premise'], x['hypothesis']) for x in self.dataset[lang][split]]
            for line in tqdm(self.dataset[lang][split]):
                item = (line['premise'], line['hypothesis'])
                inputs = tokenizer.encode(item, return_tensors="pt")
                # print(tokenizer.convert_ids_to_tokens(inputs[0]))
                tokens.append(len(inputs[0]))
            tmax = max(tokens)
            tmin = min(tokens)
            tmean = mean(tokens)
            stdev = pstdev(tokens)
            tokens_length[split] = {"max": tmax, "min": tmin, "mean": tmean, "stdev": stdev}
            print(f'{lang}-{split}: {tmax} - {tmin} - {tmean} - {stdev}')
        return tokens_length

    def count_labels(self, lang: str) -> dict:
        labels_total = {}
        for split in self.split_names:
            labels = {'0': 0, '1': 0, '2': 0}
            # feats = [(x['premise'], x['hypothesis']) for x in self.dataset[lang][split]]
            for line in tqdm(self.dataset[lang][split]):
                label = str(line['label'])
                labels[label] += 1

            labels_total[split] = labels
            print(f'{lang}-{split}: {labels}')
        return labels_total

    def check_stats(self, checkpoint: Optional[str] = None) -> None:
        """Method"""
        do_general = False
        do_tokens = False

        general_file_name = f'../../data/{self.dataset_name}_statistics.json'
        general_stats = {"dataset": self.dataset_name}
        if not os.path.isfile(general_file_name):
            do_general = True

        if checkpoint is not None:
            model_name = checkpoint.split('/')[-1]
            token_file_name = f'../../data/{self.dataset_name}_statistics_{model_name}.json'
            token_stats = {"tokenizer": checkpoint}
            if not os.path.isfile(token_file_name):
                do_tokens = True
                tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir=CACHE_DIR)

        if do_general or do_tokens:
            for lang in self.languages:
                self.dataset[lang] = load_dataset(self.dataset_name, lang, cache_dir=CACHE_DIR)
                if do_general:
                    general_stats[lang] = self.count_labels(lang)
                if do_tokens:
                    token_stats[lang] = self.count_max_token(lang, tokenizer=tokenizer)
            if do_general:
                self.save_general_stats(general_stats)
            if do_tokens:
                self.save_token_stats(checkpoint, token_stats)

    def save_general_stats(self, results: dict) -> None:
        """Method"""
        file_name = f'../../data/{self.dataset_name}_statistics.json'
        print(results)
        with open(file_name, 'w', encoding='utf-8') as outfile:
            json.dump(results, outfile)

    def save_token_stats(self, checkpoint: str, results: dict) -> None:
        """Method"""
        model_name = checkpoint.split('/')[-1]
        model_name = '-'.join(model_name.split('-')[:-1])
        file_name = f'../../data/{self.dataset_name}_statistics_{model_name}.json'
        print(results)
        with open(file_name, 'w', encoding='utf-8') as outfile:
            json.dump(results, outfile)


if __name__ == '__main__':
    dataset = 'americas_nli'
    xnli = NLIAnalyzer(dataset)
    checkpoints = [
        # "xlm-roberta-base",
        # "bigscience/bloomz-560m",
        # "bigscience/bloomz-3b",
        "bigscience/mt0-large",
    ]
    if checkpoints:
        for check in checkpoints:
            xnli.check_stats(check)
    else:
        xnli.check_stats()
