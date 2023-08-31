"""MD"""
import argparse
import json
import os
import platform
from random import sample
from itertools import chain
import torch
import numpy as np

from pytorch_lightning import seed_everything
from transformers import AutoTokenizer, \
    AutoModelForCausalLM, AutoModelForSeq2SeqLM
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


def parse_args():
    """Doc"""
    parser = argparse.ArgumentParser(
        description='This is a pipeline to test templates for the best english prompts.'
    )
    parser.add_argument(
        '--config', type=str,
        help='The experiment config file (expected in json format).',
        required=True
    )
    return parser.parse_args()


class EvaluatePipeline:
    """Doc"""
    def __init__(
            self, dataset_name: str, language: str, split: str,
            model_name: str, template_name: str, template_language: str,
            target_path: str, model=None, multiplier: int = 0,
            randomize_example: bool = False
    ):
        self.dataset_name: str = dataset_name
        self.language: str = language
        self.split: str = split
        self.model_name: str = model_name
        if 'bloomz' in self.model_name:
            self.model_family = 'bloomz'
        elif 'mt0' in self.model_name:
            self.model_family = 'mt0'
        else:
            raise ValueError(f'The model {model_name} is not implemented yet!')
        self.template_name: str = template_name
        self.template_language: str = template_language
        self.target_path: str = target_path
        self.template_file: str = f'{self.template_name}.jinja2'
        self.number_of_samples: int = 0
        self.labels_text: list = self.get_labels('labels.json')
        print('Loading datasets (examples)...')
        self.dataset_examples = load_dataset(
            self.dataset_name, self.language, cache_dir=CACHE_DIR, split='validation'
        )
        print('Loading datasets (samples)...')
        self.dataset_samples = load_dataset(
            self.dataset_name, self.language, cache_dir=CACHE_DIR, split=self.split
        )
        print('Initializing tokenizer...')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=CACHE_DIR)
        print('Initializing model...')
        self.model = None
        if model is not None:
            self.model = model
        else:
            if self.model_family == 'bloomz':
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, cache_dir=CACHE_DIR
                )
            elif self.model_family == 'mt0':
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name, cache_dir=CACHE_DIR
                )
        self.multiplier = multiplier
        self.randomize_example = randomize_example
        self.class_num = 0
        self.examples = self.get_examples()
        self.samples = self.get_samples()
        self.prompts = self.get_prompts()
        if self.class_num == 0:
            raise ValueError(
                'self.class_num has not been updated in self.get_examples(). Please check.')
        self.acc_macro = Accuracy(
            task="multiclass", num_classes=self.class_num, average='macro')
        # self.acc_micro =
        # Accuracy(task="multiclass", num_classes=self.class_num, average='micro')
        self.prec_macro = Precision(
            task="multiclass", num_classes=self.class_num, average='macro')
        # self.prec_micro =
        # Precision(task="multiclass", num_classes=self.class_num, average='micro')
        self.rec_macro = Recall(
            task="multiclass", num_classes=self.class_num, average='macro')
        # self.rec_micro =
        # Recall(task="multiclass", num_classes=self.class_num, average='micro')
        self.f1_macro = F1Score(
            task="multiclass", num_classes=self.class_num, average='macro')
        self.f1_micro = F1Score(
            task="multiclass", num_classes=self.class_num, average='micro')
        self.result_text = ''

    def get_labels(self, label_file):
        """Doc"""
        with open(label_file, 'r', encoding='utf-8') as file_handler:
            data = json.load(file_handler)
        all_labels = data[self.dataset_name][self.language]['labels']
        labels = [[x] for x in all_labels]
        return labels

    def get_examples(self):
        """Doc"""
        print('Getting examples...')
        selector = XNLIExampleSelector(
            model=self.model_name, language=self.language,
            dataset=self.dataset_examples, tokenizer=self.tokenizer
        )
        examples = selector.get_examples(self.multiplier, self.randomize_example)
        self.class_num = selector.class_num
        return examples

    def get_samples(self, parts=1.0) -> list:
        """Doc"""
        print('Getting samples...')
        selector = XNLISampleSelector(
            model=self.model_name, language=self.language, split=self.split,
            dataset=self.dataset_samples, tokenizer=self.tokenizer)
        samples = selector.get_samples(parts=parts)
        return samples

    def get_prompts(self):
        """Doc"""
        print('Preparing prompts...')
        prompter = XNLIPromptGenerator(
            language=self.language, examples=self.examples, samples=self.samples,
            template_file=self.template_file, template_language=self.template_language)
        zero_prompts, few_prompts = prompter.create_prompts()
        return zero_prompts, few_prompts

    def write_to_file(self, results: tuple, seed):
        """Doc"""
        result_text = f'Results (macro) for Seed: {seed}\n' \
                      f'Accuracy: {results[0]:.4f} - ' \
                      f'Precision: {results[1]:.4f} - ' \
                      f'Recall: {results[2]:.4f} - ' \
                      f'F1 Score: {results[3]:.4f} - ' \
                      f'Micro: {results[4]:.4f}\n' \
                      f'{results[0]:.4f}\t{results[1]:.4f}\t' \
                      f'{results[2]:.4f}\t{results[3]:.4f}\t{results[4]:.4f}\n'
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
        """Doc"""
        print(f'Results (macro) for Seed: {seed}\n'
              f'Accuracy: {results[0]:.4f} - '
              f'Precision: {results[1]:.4f} - '
              f'Recall: {results[2]:.4f} - '
              f'F1 Score: {results[3]:.4f} - '
              f'Micro Score: {results[4]:.4f}')

    def get_scores(self, preds, gold, seed, logging=True, write_to_file=True):
        """Doc"""
        accmac_score = self.acc_macro(torch.tensor(preds), torch.tensor(gold))
        # accmic_score = self.acc_micro(torch.tensor(preds), torch.tensor(gold))
        precmac_score = self.prec_macro(torch.tensor(preds), torch.tensor(gold))
        # precmic_score = self.prec_micro(torch.tensor(preds), torch.tensor(gold))
        recmac_score = self.rec_macro(torch.tensor(preds), torch.tensor(gold))
        # recmic_score = self.rec_micro(torch.tensor(preds), torch.tensor(gold))
        f1mac_score = self.f1_macro(torch.tensor(preds), torch.tensor(gold))
        f1mic_score = self.f1_micro(torch.tensor(preds), torch.tensor(gold))

        results = (
            accmac_score.cpu().numpy().tolist(),
            precmac_score.cpu().numpy().tolist(),
            recmac_score.cpu().numpy().tolist(),
            f1mac_score.cpu().numpy().tolist(),
            f1mic_score.cpu().numpy().tolist()
        )

        if logging:
            self.print_log(results=results, seed=seed)

        if write_to_file:
            self.write_to_file(results=results, seed=seed)

        return results

    def test_prompt(self, max_samples, fewshot, seed):
        if 0 < max_samples < len(self.samples):
            self.number_of_samples = max_samples
        else:
            self.number_of_samples = len(self.samples)

        # forced_labels = list(chain.from_iterable([x for x in self.labels_text]))
        forced_labels = ['Yes', 'Maybe', 'No']
        # max_answer_length = max([len(x) for x in forced_labels])
        max_answer_length = 1

        gold_labels = [x[2] for x in self.samples[:self.number_of_samples]]

        if fewshot:
            prompts = self.prompts[1][:self.number_of_samples]
        else:
            prompts = self.prompts[0][:self.number_of_samples]

        self.model.to(DEVICE)
        self.model.eval()

        prediction = []
        raw_answers = []
        for item in tqdm(prompts):
            inputs = self.tokenizer.encode(item, return_tensors="pt")
            force_words_ids = [
                self.tokenizer(forced_labels, add_special_tokens=False).input_ids,
            ]
            # max_answer_length = max([len(x) for x in force_words_ids[0]])
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    force_words_ids=force_words_ids,
                    num_beams=3,
                    num_return_sequences=1,
                    no_repeat_ngram_size=1,
                    remove_invalid_values=True,
                    max_new_tokens=max_answer_length
                )
                if self.model_family == 'bloomz':
                    result = self.tokenizer.decode(outputs[0][-1])
                    # index = result.rfind('?') + 2
                    # answer = result[index:-4]
                    # index = result.rfind('\n')
                    # answer = result[index:].strip()
                    answer = result

                elif self.model_family == 'mt0':
                    result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    answer = result

                answer_idx = -1
                # for i, labels in enumerate(self.labels_text):
                #     labels_lower = [x.lower() for x in labels]
                #     if answer.lower() in labels_lower:
                #         answer_idx = i
                #         break
                labels_lower = [x.lower() for x in forced_labels]
                if answer.lower() in labels_lower:
                    answer_idx = labels_lower.index(answer.lower())

                # All unknown answers will be regarded as a "No" or "False"
                if answer_idx == -1:
                    answer_idx = 2
                    # raise ValueError(f'The answer: {answer} can not be found in label list')

                prediction.append(answer_idx)
                raw_answers.append(answer)

        scores = self.get_scores(preds=prediction, gold=gold_labels, seed=seed)
        return_dict = {
            'prediction': prediction, 'gold_labels': gold_labels,
            'raw_answers': raw_answers, 'scores': scores,
            'prompt_example': prompts[0]
        }
        return return_dict

    def load_prediction_from_file(self, seed_number):
        result = get_prediction_from_file(self.target_path, seed_number)
        # prediction = result['prediction']
        # gold_labels = result['gold_labels']
        # scores = self.get_scores(preds=prediction, gold=gold_labels, seed=seed_number)
        # return prediction, gold_labels, scores
        return result


def get_average(n, result, result_path, seeds, previous_text):
    # stacked_result = torch.stack([torch.stack(x) for x in result])
    # torch.mean(stacked_result, dim=0)
    # -> result = single tensor with 8 elements (the average of each column)
    zipped_results = zip(*result)
    stacked_zip = [np.stack(x) for x in zipped_results]
    avg = [np.mean(x) for x in stacked_zip]
    print(f'Average Results (macro) - Samples: {n} - '
          f'Accuracy: {avg[0]:.4f} - '
          f'Precision: {avg[1]:.4f} - '
          f'Recall: {avg[2]:.4f} - '
          f'F1: {avg[3]:.4f} - '
          f'Micro: {avg[4]:.4f}')

    file_name = f'{result_path}.txt'

    with open(file_name, 'w', encoding='utf-8') as file_handler:
        print(previous_text, file=file_handler)
        print(f'Average Results (macro) - Samples: {n} - Seeds: {seeds}\n'
              f'Accuracy   : {avg[0]:.4f}\n'
              f'Precision  : {avg[1]:.4f}\n'
              f'Recall     : {avg[2]:.4f}\n'
              f'F1 Score   : {avg[3]:.4f}\n'
              f'Micro Score: {avg[4]:.4f}\n'
              f'{avg[0]:.4f}\t{avg[1]:.4f}\t'
              f'{avg[2]:.4f}\t{avg[3]:.4f}\t{avg[4]:.4f}', file=file_handler)


def get_result_file_path(
        dataset_name, model_name, fewshots, target_path, template_name,
        language, template_language, multiplier, random
):
    model = model_name.split('/')[-1]
    if fewshots:
        qty = multiplier * 3
        shot = f'few{qty}'
        if random:
            shot += '_random'
    else:
        shot = 'zero'
    file_name = \
        f'{target_path}/{dataset_name}_{model}_{template_name}_' \
        f'{language}_{template_language}_{shot}'
    return file_name


def read_experiment_config(file_name: str) -> dict:
    with open(file_name, 'r', encoding='utf-8') as cfile:
        data = json.load(cfile)
    return data


def write_prediction_to_file(result_dict, result_path, seed_number):
    file_name = f'{result_path}_{seed_number}.json'
    # prediction = result_tuple[0]
    # gold_labels = result_tuple[1]
    # result_dict = {'prediction': prediction, 'gold_labels': gold_labels}
    with open(file_name, 'w', encoding='utf-8') as file_handler:
        json.dump(result_dict, file_handler, indent=4)


def get_prediction_from_file(result_path, seed):
    file_name = f'{result_path}_{seed}.json'
    with open(file_name, 'r', encoding='utf-8') as file_handler:
        data = json.load(file_handler)
    # pred = data['prediction']
    # gold = data['gold_labels']
    return data


if __name__ == '__main__':
    args = parse_args()
    params = read_experiment_config(args.config)

    local_test = False

    # Hyperparams
    DATASET_NAME = params['dataset_name']  # xnli'
    LANGUAGES = params['languages']  # en'
    SAMPLE_SPLIT = params['sample_split']  # validation
    MODEL_NAME = params['model_name']  # 'bigscience/bloomz-560m'
    TEMPLATE_NAME = params['template_name']  # 'nli_english'
    TEMPLATE_LANGUAGE = params['template_language']
    TARGET_PATH = params['target_path']  # '../experiments/best_english_prompts'
    # Checking if the folder TARGET_PATH already exist or not.
    # If folder not exist, then create it.
    if not os.path.exists(TARGET_PATH):
        os.makedirs(TARGET_PATH)
    FEWSHOT = params['fewshot']  # True
    MULTIPLIER = params['multiplier']
    RANDOMIZE_EXAMPLE = params['randomize_example']
    MAX_ANSWER_LENGTH = params['max_answer_length']
    MAX_SAMPLES = params['max_samples']
    RANDOMIZE_SEEDS = params['randomize_seeds']
    NUMBER_OF_SEEDS = params['number_of_seeds']
    SEEDS = params['seeds']
    if local_test:
        LANGUAGES = ['en']
        # MAX_SAMPLES = 5
        RANDOMIZE_SEEDS = False

    # Preloading model in advance
    print(f'Preloading model {MODEL_NAME}...')
    MODEL = None
    if 'bloomz' in MODEL_NAME:
        MODEL = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    elif 'mt0' in MODEL_NAME:
        MODEL = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)

    if RANDOMIZE_SEEDS:
        seed_pool = list(range(10000))
        SEEDS = sample(seed_pool, NUMBER_OF_SEEDS)
    # else:
        # SEEDS = [5029, 9283, 5672, 3345, 2055]
        # SEEDS = [2613, 3695, 6444, 7170, 9944]
        # SEEDS = [9888]
        # LANGUAGES = ['en']

    for lang in LANGUAGES:
        if TEMPLATE_LANGUAGE == 'default':
            tpl_language = lang
        else:
            tpl_language = TEMPLATE_LANGUAGE

        result_file_path = get_result_file_path(
            DATASET_NAME, MODEL_NAME, FEWSHOT, TARGET_PATH,
            TEMPLATE_NAME, lang, tpl_language, MULTIPLIER,
            RANDOMIZE_EXAMPLE
        )

        res = []
        result_file_name = f'{result_file_path}.txt'
        if os.path.isfile(result_file_name):
            # os.remove(result_file_path)
            continue
        TEXT_HOLDER = ''
        for seed in SEEDS:
            seed_everything(seed)
            print('Initializing examples, samples, and prompts...')
            pipe = EvaluatePipeline(
                DATASET_NAME, lang, SAMPLE_SPLIT, MODEL_NAME,
                TEMPLATE_NAME, tpl_language, result_file_path, MODEL, MULTIPLIER,
                RANDOMIZE_EXAMPLE
            )
            print('Testing prompts...')
            prediction_file_name = f'{result_file_path}_{seed}.json'
            if os.path.isfile(prediction_file_name):
                results = pipe.load_prediction_from_file(seed)
            else:
                results = pipe.test_prompt(
                    MAX_SAMPLES, FEWSHOT, seed=seed)
            res.append(results['scores'])
            write_prediction_to_file(results, result_file_path, seed)
            TEXT_HOLDER += pipe.result_text
            total = pipe.number_of_samples
        # process_statistics(result_file_path, SEEDS, res, TEXT_HOLDER)

        get_average(total, res, result_file_path, SEEDS, TEXT_HOLDER)
