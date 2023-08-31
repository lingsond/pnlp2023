"""MD"""
import argparse
import json
import platform
from random import sample
from pytorch_lightning import seed_everything
import torch
from peft_xnli import PEFTTrainer
from preprocess_peft_xnli import XNLIPEFTPreprocessor


if platform.system() == 'Windows':
    CACHE_DIR = 'D:/Cache/huggingface'
else:
    CACHE_DIR = '/home/stud/wangsadirdja/.cache/huggingface/'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# if DEVICE == 'cuda':
#     torch.set_default_dtype(torch.cuda.FloatTensor)
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


if __name__ == '__main__':
    args = parse_args()
    params = read_experiment_config(args.config)

    # PATHS
    PATH_TRAIN = params['json_train']
    PATH_EVAL = params['json_eval']
    PATH_TEST = params['json_test']
    # HYPERPARAMS
    DATASET_NAME = params['dataset_name']  # xnli'
    DATASET_SIZE = params['dataset_size']
    # Get the model's info
    MODEL_NAME = params['model_name']  # 'bigscience/bloomz-560m'
    # model_parts = MODEL_NAME.split('/')
    # model_parts = model_parts[-1].split('-')
    # model_family = model_parts[0]
    # model_params = model_parts[-1]
    PEFT_TYPE = params['peft_type']
    MAX_ANSWER_LENGTH = params['max_answer_length']
    MAX_LENGTH = params['max_length']
    LEARNING_RATE = params['lr']
    NUM_EPOCHS = params['epochs']
    BATCH_SIZE = params['batch_size']
    VIRTUAL_TOKEN = params['virtual_token']
    RANDOMIZE_SEEDS = params['randomize_seeds']
    SEEDS = params['seeds']

    if RANDOMIZE_SEEDS:
        seed_pool = list(range(10000))
        SEEDS = sample(seed_pool, 1)
    seed_everything(SEEDS)

    # Initialization
    print(f'Preparing PEFT model for {MODEL_NAME}...')
    if DATASET_NAME == 'xnli':
        preprocessor = XNLIPEFTPreprocessor(
            model_name=MODEL_NAME, max_length=MAX_LENGTH, batch_size=BATCH_SIZE,
            train_json=PATH_TRAIN, eval_json=PATH_EVAL, test_json=PATH_TEST
        )
    else:
        raise NotImplementedError(
            f'The training pipeline for dataset {DATASET_NAME} has not '
            f'been implemented yet.'
        )
    peft = PEFTTrainer(
        MODEL_NAME, MAX_LENGTH, LEARNING_RATE, NUM_EPOCHS,
        BATCH_SIZE, VIRTUAL_TOKEN, preprocessor, PEFT_TYPE, DATASET_SIZE
    )

    peft.train()
