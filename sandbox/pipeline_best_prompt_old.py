"""MD"""
import json
import torch
import os
from pytorch_lightning import seed_everything
from transformers import AutoModelForCausalLM, AutoTokenizer
from example_selector import XNLIExampleSelector
from sample_selector import XNLISampleSelector
from prompt_generator import XNLIPromptGenerator
from torchmetrics import Accuracy, Precision, Recall, F1Score
from tqdm import tqdm


CACHE_DIR = 'D:/Cache/huggingface'


def get_labels(dataset, lang):
    with open('labels.json', 'r', encoding='utf-8') as file_handler:
        data = json.load(file_handler)
    labels = data[dataset][lang]
    return labels


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL, cache_dir=CACHE_DIR)
    lmodel = AutoModelForCausalLM.from_pretrained(MODEL, cache_dir=CACHE_DIR)
    labels_text = get_labels(dataset=DATASET, lang=LANGUAGE)
    e_selector = XNLIExampleSelector(model=MODEL, language=LANGUAGE)
    examples = e_selector.get_examples()
    s_selector = XNLISampleSelector(model=MODEL, language=LANGUAGE, split=SPLIT)
    samples = s_selector.get_samples(parts=0.1)

    if 0 < MAX_SAMPLES <= len(samples):
        n = MAX_SAMPLES
    else:
        n = len(samples)
    gold_labels = [x[2] for x in samples[:n]]

    template_file = f'./templates/{TEMPLATE}.jinja2'
    prompter = XNLIPromptGenerator(
        language=LANGUAGE, examples=examples, samples=samples,
        template_file=template_file)
    zero_prompts, few_prompts = prompter.create_prompts()
    if FEWSHOT:
        prompts = few_prompts[:n]
    else:
        prompts = zero_prompts[:n]

    prediction = []

    for item in tqdm(prompts):
        inputs = tokenizer.encode(item, return_tensors="pt")
        outputs = lmodel.generate(inputs, max_new_tokens=2048)
        result = tokenizer.decode(outputs[0])
        index = result.rfind('?') + 2
        answer = result[index:-4]
        answer_idx = labels_text.index(answer)
        prediction.append(answer_idx)

    acc_macro = Accuracy(task="multiclass", num_classes=3, average='macro')
    acc_micro = Accuracy(task="multiclass", num_classes=3, average='micro')
    prec_macro = Precision(task="multiclass", num_classes=3, average='macro')
    prec_micro = Precision(task="multiclass", num_classes=3, average='micro')
    rec_macro = Recall(task="multiclass", num_classes=3, average='macro')
    rec_micro = Recall(task="multiclass", num_classes=3, average='micro')
    f1_macro = F1Score(task="multiclass", num_classes=3, average='macro')
    f1_micro = F1Score(task="multiclass", num_classes=3, average='micro')

    accmac_score = acc_macro(torch.tensor(prediction), torch.tensor(gold_labels))
    accmic_score = acc_micro(torch.tensor(prediction), torch.tensor(gold_labels))
    precmac_score = prec_macro(torch.tensor(prediction), torch.tensor(gold_labels))
    precmic_score = prec_micro(torch.tensor(prediction), torch.tensor(gold_labels))
    recmac_score = rec_macro(torch.tensor(prediction), torch.tensor(gold_labels))
    recmic_score = rec_micro(torch.tensor(prediction), torch.tensor(gold_labels))
    f1mac_score = f1_macro(torch.tensor(prediction), torch.tensor(gold_labels))
    f1mic_score = f1_micro(torch.tensor(prediction), torch.tensor(gold_labels))

    results = (accmic_score, accmac_score, precmic_score, precmac_score, recmic_score, recmac_score, f1mic_score, f1mac_score)

    print(f'Samples: {n} - '
          f'Acc (micro/macro): {accmic_score:.4f}/{accmac_score:.4f} - '
          f'Precision: {precmic_score:.4f}/{precmac_score:.4f} - '
          f'Recall: {recmic_score:.4f}/{recmac_score:.4f} - '
          f'F1: {f1mic_score:.4f}/{f1mac_score:.4f}')

    with open(RESULT_FILE, 'a', encoding='utf-8') as file_handler:
        print(f'Results (micro/macro) for Seed: {SEEDS[idx]}\n'
              f'Accuracy: {accmic_score:.4f}/{accmac_score:.4f} - '
              f'Precision: {precmic_score:.4f}/{precmac_score:.4f} - '
              f'Recall: {recmic_score:.4f}/{recmac_score:.4f} - '
              f'F1 Score: {f1mic_score:.4f}/{f1mac_score:.4f}', file=file_handler)

    return n, results


def get_result_file_path():
    model = MODEL.split('/')[-1]
    if FEWSHOT:
        shot = 'few'
    else:
        shot = 'zero'
    file_name = f'{TARGET_PATH}/{DATASET}_{model}_{TEMPLATE}_{shot}.txt'
    return file_name


def get_average(n, result):
    # stacked_result = torch.stack([torch.stack(x) for x in result])
    # torch.mean(stacked_result, dim=0) -> result = single tensor with 8 elements (the average of each column)
    zipped_results = zip(*result)
    stacked_zip = [torch.stack(x) for x in zipped_results]
    avg = [torch.mean(x) for x in stacked_zip]
    print(f'Average Results - Samples: {n} - '
          f'Acc (micro/macro): {avg[0]:.4f}/{avg[1]:.4f} - '
          f'Precision: {avg[2]:.4f}/{avg[3]:.4f} - '
          f'Recall: {avg[4]:.4f}/{avg[5]:.4f} - '
          f'F1: {avg[6]:.4f}/{avg[7]:.4f}')

    with open(RESULT_FILE, 'a', encoding='utf-8') as file_handler:
        print(f'Average Results (micro/macro) - Samples: {n} - Seeds: {SEEDS}\n'
              f'Accuracy : {avg[0]:.4f}/{avg[1]:.4f}\n'
              f'Precision: {avg[2]:.4f}/{avg[3]:.4f}\n'
              f'Recall   : {avg[4]:.4f}/{avg[5]:.4f}\n'
              f'F1 Score : {avg[6]:.4f}/{avg[7]:.4f}', file=file_handler)


if __name__ == '__main__':
    DATASET = 'xnli'
    LANGUAGE = 'en'
    SPLIT = 'test'
    TEMPLATE = 'english01'
    MODEL = 'bigscience/bloomz-560m'
    TARGET_PATH = '../experiments/best_english_prompts'
    FEWSHOT = False
    MAX_SAMPLES = 10
    SEEDS = [42, 7, 73, 5040, 100]
    res = []
    RESULT_FILE = get_result_file_path()
    if os.path.isfile(RESULT_FILE):
        os.remove(RESULT_FILE)
    for idx, seed in enumerate(SEEDS):
        seed_everything(seed)
        total, stats = main()
        res.append(stats)
    get_average(total, res)
