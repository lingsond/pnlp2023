"""MD"""
import json
import os
import platform
import torch
from random import sample
from pytorch_lightning import seed_everything
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from torchmetrics import Accuracy, Precision, Recall, F1Score
from datasets import load_dataset
from tqdm import tqdm
from example_selector import XNLIExampleSelector
from sample_selector import XNLISampleSelector
from prompt_generator import XNLIPromptGenerator

if platform.system() == 'Windows':
    CACHE_DIR = 'D:/Cache/huggingface'
else:
    CACHE_DIR = '/home/stud/wangsadirdja/.cache/huggingface/'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cuda':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.set_default_device(DEVICE)


class BestPromptPipeline:
    """Doc"""
    def __init__(
            self, dataset_name: str, language: str, split: str, model_name: str,
            template_name: str, target_path: str
    ):
        self.dataset_name: str = dataset_name
        self.language: str = language
        self.split: str = split
        self.model_name: str = model_name
        self.template_name: str = template_name
        self.target_path: str = target_path
        self.template_file: str = f'{self.template_name}.jinja2'
        self.number_of_samples: int = 0
        self.labels_text: list = self.get_labels()
        self.dataset_examples = load_dataset(self.dataset_name, self.language, cache_dir=CACHE_DIR, split='validation')
        self.dataset_samples = load_dataset(self.dataset_name, self.language, cache_dir=CACHE_DIR, split=self.split)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=CACHE_DIR)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, cache_dir=CACHE_DIR)
        self.class_num = 0
        self.examples = self.get_examples()
        self.samples = self.get_samples(parts=0.1)
        self.prompts = self.get_prompts()
        if self.class_num == 0:
            raise ValueError(
                'self.class_num has not been updated in self.get_examples(). Please check.')
        self.acc_macro = Accuracy(task="multiclass", num_classes=self.class_num, average='macro')
        # self.acc_micro = Accuracy(task="multiclass", num_classes=self.class_num, average='micro')
        self.prec_macro = Precision(task="multiclass", num_classes=self.class_num, average='macro')
        # self.prec_micro = Precision(task="multiclass", num_classes=self.class_num, average='micro')
        self.rec_macro = Recall(task="multiclass", num_classes=self.class_num, average='macro')
        # self.rec_micro = Recall(task="multiclass", num_classes=self.class_num, average='micro')
        self.f1_macro = F1Score(task="multiclass", num_classes=self.class_num, average='macro')
        self.f1_micro = F1Score(task="multiclass", num_classes=self.class_num, average='micro')
        self.result_text = ''

    def get_labels(self):
        with open('labels.json', 'r', encoding='utf-8') as file_handler:
            data = json.load(file_handler)
        labels = data[self.dataset_name][self.language]
        return labels

    def get_examples(self):
        selector = XNLIExampleSelector(
            model=self.model_name, language=self.language,
            dataset=self.dataset_examples, tokenizer=self.tokenizer
        )
        examples = selector.get_examples()
        self.class_num = selector.class_num
        return examples

    def get_samples(self, parts=0.1) -> list:
        selector = XNLISampleSelector(
            model=self.model_name, language=self.language, split=self.split,
            dataset=self.dataset_samples, tokenizer=self.tokenizer)
        samples = selector.get_samples(parts=parts)
        return samples

    def get_prompts(self):
        prompter = XNLIPromptGenerator(
            language=self.language, examples=self.examples, samples=self.samples,
            template_file=self.template_file)
        zero_prompts, few_prompts = prompter.create_prompts()
        return zero_prompts, few_prompts

    def write_to_file(self, results: tuple, seed):
        result_text = f'Results (macro) for Seed: {seed}\n' \
                      f'Accuracy: {results[0]:.4f} - ' \
                      f'Precision: {results[1]:.4f} - ' \
                      f'Recall: {results[2]:.4f} - ' \
                      f'F1 Score: {results[3]:.4f} - ' \
                      f'Micro: {results[4]:.4f}\n'
        self.result_text += result_text
        # with open(self.target_path, 'a', encoding='utf-8') as file_handler:
        #     print(f'Results (macro) for Seed: {seed}\n'
        #           f'Accuracy: {results[0]:.4f} - '
        #           f'Precision: {results[1]:.4f} - '
        #           f'Recall: {results[2]:.4f} - '
        #           f'F1 Score: {results[3]:.4f} - '
        #           f'Micro: {results[4]:.4f}', file=file_handler)

    @staticmethod
    def print_log(results: tuple, seed):
        print(f'Results (macro) for Seed: {seed}\n'
              f'Accuracy: {results[0]:.4f} - '
              f'Precision: {results[1]:.4f} - '
              f'Recall: {results[2]:.4f} - '
              f'F1 Score: {results[3]:.4f} - '
              f'Micro Score: {results[4]:.4f}')

    def get_scores(self, preds, gold, seed, logging=True, write_to_file=True):
        accmac_score = self.acc_macro(torch.tensor(preds), torch.tensor(gold))
        # accmic_score = self.acc_micro(torch.tensor(preds), torch.tensor(gold))
        precmac_score = self.prec_macro(torch.tensor(preds), torch.tensor(gold))
        # precmic_score = self.prec_micro(torch.tensor(preds), torch.tensor(gold))
        recmac_score = self.rec_macro(torch.tensor(preds), torch.tensor(gold))
        # recmic_score = self.rec_micro(torch.tensor(preds), torch.tensor(gold))
        f1mac_score = self.f1_macro(torch.tensor(preds), torch.tensor(gold))
        f1mic_score = self.f1_micro(torch.tensor(preds), torch.tensor(gold))

        results = (
            accmac_score, precmac_score, recmac_score, f1mac_score,
            f1mic_score)

        if logging:
            self.print_log(results=results, seed=seed)

        if write_to_file:
            self.write_to_file(results=results, seed=seed)

        return results

    def test_prompt(self, max_samples, fewshot, seed, max_seq_length=32):
        if 0 < max_samples < len(self.samples):
            self.number_of_samples = max_samples
        else:
            self.number_of_samples = len(self.samples)

        gold_labels = [x[2] for x in self.samples[:self.number_of_samples]]

        if fewshot:
            prompts = self.prompts[1][:self.number_of_samples]
        else:
            prompts = self.prompts[0][:self.number_of_samples]

        prediction = []

        for item in tqdm(prompts):
            inputs = self.tokenizer.encode(item, return_tensors="pt")
            outputs = self.model.generate(inputs, max_new_tokens=max_seq_length)
            result = self.tokenizer.decode(outputs[0][-2])
            # index = result.rfind('?') + 2
            # answer = result[index:-4]
            answer_idx = self.labels_text.index(result)
            prediction.append(answer_idx)

        scores = self.get_scores(preds=prediction, gold=gold_labels, seed=seed)
        return scores


def get_average(n, result, result_path, seeds, previous_text):
    # stacked_result = torch.stack([torch.stack(x) for x in result])
    # torch.mean(stacked_result, dim=0) -> result = single tensor with 8 elements (the average of each column)
    zipped_results = zip(*result)
    stacked_zip = [torch.stack(x) for x in zipped_results]
    avg = [torch.mean(x) for x in stacked_zip]
    print(f'Average Results (macro) - Samples: {n} - '
          f'Accuracy: {avg[0]:.4f} - '
          f'Precision: {avg[1]:.4f} - '
          f'Recall: {avg[2]:.4f} - '
          f'F1: {avg[3]:.4f} - '
          f'Micro: {avg[4]:.4f}')

    with open(result_path, 'w', encoding='utf-8') as file_handler:
        print(previous_text, file=file_handler)
        print(f'Average Results (macro) - Samples: {n} - Seeds: {seeds}\n'
              f'Accuracy   : {avg[0]:.4f}\n'
              f'Precision  : {avg[1]:.4f}\n'
              f'Recall     : {avg[2]:.4f}\n'
              f'F1 Score   : {avg[3]:.4f}\n'
              f'Micro Score: {avg[4]:.4f}', file=file_handler)


def get_result_file_path(dataset_name, model_name, fewshots, target_path, template_name):
    model = model_name.split('/')[-1]
    if fewshots:
        shot = 'few'
    else:
        shot = 'zero'
    file_name = f'{target_path}/{dataset_name}_{model}_{template_name}_{shot}.txt'
    return file_name


if __name__ == '__main__':
    # Hyperparams
    DATASET_NAME = 'xnli'
    LANGUAGE = 'en'
    SAMPLE_SPLIT = 'validation'
    MODEL_NAME = 'bigscience/mt0-base'
    TEMPLATE_BASE = 'nli_english'
    TARGET_PATH = '../experiments/best_english_prompts'
    FEWSHOT = False
    MAX_ANSWER_LENGTH = 20
    MAX_SAMPLES = 0
    RANDOMIZE_SEEDS = True

    if RANDOMIZE_SEEDS:
        seed_pool = list(range(10000))
        SEEDS = sample(seed_pool, 5)
    else:
        # SEEDS = [42, 7, 73, 5040, 100]
        SEEDS = [42]

    for idx in range(1, 11):
        number = str(idx).zfill(2)
        TEMPLATE_NAME = f'{TEMPLATE_BASE}{number}'

        result_file_path = get_result_file_path(
            DATASET_NAME, MODEL_NAME, FEWSHOT, TARGET_PATH, TEMPLATE_NAME)

        res = []
        if os.path.isfile(result_file_path):
            # os.remove(result_file_path)
            continue
        TEXT_HOLDER = ''
        for seed in SEEDS:
            seed_everything(seed)
            print('Initializing examples, samples, and prompts...')
            pipe = BestPromptPipeline(
                DATASET_NAME, LANGUAGE, SAMPLE_SPLIT, MODEL_NAME,
                TEMPLATE_NAME, result_file_path)
            print('Testing prompts...')
            stats = pipe.test_prompt(
                MAX_SAMPLES, FEWSHOT, seed=seed, max_seq_length=MAX_ANSWER_LENGTH)
            res.append(stats)
            TEXT_HOLDER += pipe.result_text
            total = pipe.number_of_samples
        get_average(total, res, result_file_path, SEEDS, TEXT_HOLDER)
