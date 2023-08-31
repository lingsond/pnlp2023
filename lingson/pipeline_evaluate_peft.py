# IMPORTS
# Standard Libraries
import argparse
import json
import os
import platform
from tqdm import tqdm
from random import sample
from itertools import chain
# Other Libraries
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM,\
    AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict,\
    PrefixTuningConfig, TaskType, PeftConfig, PeftModelForCausalLM, LoraConfig,\
    AdaLoraConfig, PeftModelForSeq2SeqLM
from torchmetrics import Accuracy, Precision, Recall, F1Score
from pytorch_lightning import seed_everything
# Import Project Libraries
from preprocess_peft_xnli import XNLIPEFTPreprocessor
from example_selector import XNLIExampleSelector
from sample_selector import XNLISampleSelector
from prompt_generator import XNLIPromptGenerator
from shots_generator import XNLIShotsGenerator
from peft_evaluate import PEFTEvaluator


if platform.system() == 'Windows':
    CACHE_DIR = 'D:/Cache/huggingface'
else:
    CACHE_DIR = '/home/stud/wangsadirdja/.cache/huggingface/'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_device(DEVICE)


def parse_args():
    """Doc"""
    parser = argparse.ArgumentParser(
        description='This is a pipeline to train LLM using PEFT.'
    )
    parser.add_argument(
        '--config', type=str,
        help='The experiment config file (expected in json format).',
        required=True
    )
    return parser.parse_args()


def read_experiment_config(file_name: str) -> dict:
    with open(file_name, 'r', encoding='utf-8') as cfile:
        data = json.load(cfile)
    return data


def get_result_file_path(
        peft_type, dataset_size, model_name, fewshots, target_path,
        template_name, language
):
    model = model_name.split('/')[-1]
    if fewshots > 0:
        shot = f'few{fewshots}'
    else:
        shot = 'zero'
    file_path = \
        f'{target_path}/{model}_{peft_type}{dataset_size}_{template_name}_' \
        f'{language}_{shot}'
    return file_path


def load_prediction_from_file(file_name):
    with open(file_name, 'r', encoding='utf-8') as file_handler:
        data = json.load(file_handler)
    return data


def write_prediction_to_file(result_dict, file_name):
    with open(file_name, 'w', encoding='utf-8') as file_handler:
        json.dump(result_dict, file_handler, indent=4)


def get_average(result, result_path, seeds, previous_text):
    # stacked_result = torch.stack([torch.stack(x) for x in result])
    # torch.mean(stacked_result, dim=0)
    # -> result = single tensor with 8 elements (the average of each column)
    zipped_results = zip(*result)
    stacked_zip = [np.stack(x) for x in zipped_results]
    avg = [np.mean(x) for x in stacked_zip]
    print(f'Average Results (macro) -  '
          f'Accuracy: {avg[0]:.4f} - '
          f'Precision: {avg[1]:.4f} - '
          f'Recall: {avg[2]:.4f} - '
          f'F1: {avg[3]:.4f} - '
          f'Micro: {avg[4]:.4f}')

    file_name = f'{result_path}.txt'

    with open(file_name, 'w', encoding='utf-8') as file_handler:
        print(previous_text, file=file_handler)
        print(f'Average Results (macro) - Seeds: {seeds}\n'
              f'Accuracy   : {avg[0]:.4f}\n'
              f'Precision  : {avg[1]:.4f}\n'
              f'Recall     : {avg[2]:.4f}\n'
              f'F1 Score   : {avg[3]:.4f}\n'
              f'Micro Score: {avg[4]:.4f}\n'
              f'{avg[0]:.4f}\t{avg[1]:.4f}\t'
              f'{avg[2]:.4f}\t{avg[3]:.4f}\t{avg[4]:.4f}', file=file_handler)


if __name__ == '__main__':
    args = parse_args()
    params = read_experiment_config(args.config)

    # Hyperparams
    DATASET_NAME = params['dataset_name']
    DATASET_SIZE = params['dataset_size']
    LANGUAGES = params['languages']
    TEMPLATE_NAME = params['template_name']
    template_file = f'{TEMPLATE_NAME}_prompt.jinja2'
    TARGET_PATH = params['target_path']
    # Checking if the folder TARGET_PATH already exist or not.
    # If folder not exist, then create it.
    if not os.path.exists(TARGET_PATH):
        os.makedirs(TARGET_PATH)
    MODEL_NAME = params['model_name']
    PEFT_TYPE = params['peft_type']
    PEFT_MAX_LENGTH = params['peft_max_length']
    FEWSHOTS = params['fewshots']
    MAX_SAMPLES = params['max_samples']
    RANDOMIZE_SEEDS = params['randomize_seeds']
    NUMBER_OF_SEEDS = params['number_of_seeds']
    SEEDS = params['seeds']
    if 'bloomz' in MODEL_NAME:
        model_family = 'bloomz'
    elif 'mt0' in MODEL_NAME:
        model_family = 'mt0'
    else:
        raise NotImplementedError(f'Model {MODEL_NAME} not implemented yet.')
    peft_model_id = f'{MODEL_NAME}_xnli_'
    if PEFT_TYPE == 'prefix':
        peft_model_id += 'PREFIX_TUNING_'
    elif PEFT_TYPE == 'lora':
        peft_model_id += 'LORA_'
    if model_family == 'bloomz':
        peft_model_id += 'CAUSAL_LM_'
    elif model_family == 'mt0':
        peft_model_id += 'SEQ_2_SEQ_LM_'
    peft_model_id += f'{PEFT_MAX_LENGTH}_{DATASET_SIZE}'

    if RANDOMIZE_SEEDS:
        seed_pool = list(range(10000))
        SEEDS = sample(seed_pool, NUMBER_OF_SEEDS)

    # Preloading model in advance
    print(f'Preloading model {MODEL_NAME}... and the PEFT model')
    config = PeftConfig.from_pretrained(peft_model_id)
    MODEL = None
    if 'bloomz' in MODEL_NAME:
        MODEL = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
        # model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
        PEFT_MODEL = PeftModelForCausalLM.from_pretrained(MODEL, peft_model_id)
    elif 'mt0' in MODEL_NAME:
        MODEL = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
        PEFT_MODEL = PeftModelForSeq2SeqLM.from_pretrained(MODEL, peft_model_id)
    else:
        raise NotImplementedError(f'Model {MODEL_NAME} not implemented yet.')

    for language in LANGUAGES:
        print(f'Processing language: {language}')
        # Initialization
        eval_generator = XNLIShotsGenerator(
            language=language, tpl_file=template_file, tpl_language='en',
            split='validation', n=[100]
        )
        test_generator = XNLIShotsGenerator(
            language=language, tpl_file='nli01_prompt.jinja2', tpl_language='en',
            split='test', n=[0]
        )
        eval_data = eval_generator.create_prompts()
        test_data = test_generator.create_prompts()

        result_file_path = get_result_file_path(
            PEFT_TYPE, DATASET_SIZE, MODEL_NAME, FEWSHOTS, TARGET_PATH,
            TEMPLATE_NAME, language
        )

        res = []
        result_file_name = f'{result_file_path}.txt'
        if os.path.isfile(result_file_name):
            # os.remove(result_file_path)
            continue
        TEXT_HOLDER = ''

        for seed in SEEDS:
            seed_everything(seed)

            peft = PEFTEvaluator(
                PEFT_MODEL, eval_data, test_data, MODEL_NAME, peft_model_id,
                TARGET_PATH
            )
            prediction_file_name = f'{result_file_path}_{seed}.json'
            if os.path.isfile(prediction_file_name):
                results = load_prediction_from_file(prediction_file_name)
            else:
                results = peft.eval(MAX_SAMPLES, FEWSHOTS, seed)
            res.append(results['scores'])
            write_prediction_to_file(results, prediction_file_name)
            TEXT_HOLDER += peft.result_text
            # total = peft.number_of_samples

        get_average(res, result_file_path, SEEDS, TEXT_HOLDER)

