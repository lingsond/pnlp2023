# IMPORTS
# Standard Libraries
import argparse
import json
import platform
from tqdm import tqdm
from random import sample
# Other Libraries
import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM,\
    AutoTokenizer
from peft import PeftConfig, PeftModelForCausalLM, LoraConfig,\
    AdaLoraConfig, PeftModelForSeq2SeqLM
from torchmetrics import Accuracy, Precision, Recall, F1Score
from pytorch_lightning import seed_everything
# Import Project Libraries
from preprocess_peft_xnli import XNLIPEFTPreprocessor
from example_selector import XNLIExampleSelector
from sample_selector import XNLISampleSelector
from prompt_generator import XNLIPromptGenerator
from shots_generator import XNLIShotsGenerator


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


class PEFTEvaluator:
    """This only uses English templates for the prompts"""

    def __init__(
            self, peft_model, data_eval, data_test,
            model_name, peft_model_id, target_path
    ):
        self.dataset_eval = data_eval
        self.dataset_test = data_test
        self.number_of_samples = len(data_test)
        self.model_name: str = model_name
        if 'bloomz' in self.model_name:
            self.model_family = 'bloomz'
        elif 'mt0' in self.model_name:
            self.model_family = 'mt0'
        else:
            raise ValueError(f'The model {model_name} is not implemented yet!')
        self.peft_model_id = peft_model_id
        self.peft_model = self.init_peft_model(peft_model)
        self.target_path = target_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, cache_dir=CACHE_DIR
        )
        self.labels = ['Yes', 'Maybe', 'No']
        self.result_text = ''

    def init_peft_model(self, peft_model):
        if peft_model is None:
            print('Loading PEFT model')
            config = PeftConfig.from_pretrained(self.peft_model_id)
            if self.model_family == 'bloomz':
                model = AutoModelForCausalLM.from_pretrained(
                    config.base_model_name_or_path, cache_dir=CACHE_DIR
                )
                pmodel = PeftModelForCausalLM.from_pretrained(model, self.peft_model_id)
            elif self.model_family == 'mt0':
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    config.base_model_name_or_path, cache_dir=CACHE_DIR
                )
                pmodel = PeftModelForSeq2SeqLM.from_pretrained(model, self.peft_model_id)
            else:
                raise NotImplementedError
        else:
            pmodel = peft_model
        return pmodel

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

    @staticmethod
    def print_log(results: tuple, seed):
        """Doc"""
        print(f'Results (macro) for Seed: {seed}\n'
              f'Accuracy: {results[0]:.4f} - '
              f'Precision: {results[1]:.4f} - '
              f'Recall: {results[2]:.4f} - '
              f'F1 Score: {results[3]:.4f} - '
              f'Micro Score: {results[4]:.4f}')

    def get_multi_scores(
            self, preds, gold, seed_number, logging=True, write_to_file=True
    ):
        acc_macro = Accuracy(
            task="multiclass", num_classes=3, average='macro')
        prec_macro = Precision(
            task="multiclass", num_classes=3, average='macro')
        rec_macro = Recall(
            task="multiclass", num_classes=3, average='macro')
        f1_macro = F1Score(
            task="multiclass", num_classes=3, average='macro')
        f1_micro = F1Score(
            task="multiclass", num_classes=3, average='micro')

        accmac_score = acc_macro(torch.tensor(preds), torch.tensor(gold))
        precmac_score = prec_macro(torch.tensor(preds), torch.tensor(gold))
        recmac_score = rec_macro(torch.tensor(preds), torch.tensor(gold))
        f1mac_score = f1_macro(torch.tensor(preds), torch.tensor(gold))
        f1mic_score = f1_micro(torch.tensor(preds), torch.tensor(gold))

        results = (
            accmac_score.cpu().numpy().tolist(),
            precmac_score.cpu().numpy().tolist(),
            recmac_score.cpu().numpy().tolist(),
            f1mac_score.cpu().numpy().tolist(),
            f1mic_score.cpu().numpy().tolist()
        )

        if logging:
            self.print_log(results=results, seed=seed_number)

        if write_to_file:
            self.write_to_file(results=results, seed=seed_number)

        return results

    def get_prompts(self, eval_set, test_set, shots):
        # Select 'shots' random samples from the eval set for examples
        selection = sample(eval_set, shots)
        examples = []
        for i, item in enumerate(selection):
            tokens = self.tokenizer.encode(item)
            examples.append({'text': item, 'length': len(tokens)})
        # To still get the most shots when truncation happen, the examples are sorted.
        examples.sort(key=lambda x: x['length'])
        if self.model_family == 'bloomz':
            max_token_length = 2048
        elif self.model_family == 'mt0':
            max_token_length = 1024
        else:
            raise NotImplementedError(f'Model {self.model_name} not implemented yet.')
        new_prompts = []
        for item in test_set:
            prompt_token = self.tokenizer.encode(item)
            pdict = {'text': item, 'length': len(prompt_token)}
            examples.insert(0, pdict)
            # If truncation is needed, remove shots until prompt is short enough
            prompts_length = [x['length'] for x in examples]
            i = len(examples)
            longest = sum(prompts_length) + i - 1
            while longest > max_token_length and i > 1:
                i -= 1
                longest = sum(prompts_length[:i]) + i - 1
            # Reverse list so sample is the last prompt, after all examples.
            new_examples = examples[:i][::-1]
            new_prompt = '</s>'.join([x['text'] for x in new_examples])
            # new_prompt = '\n'.join([x['text'] for x in new_examples])
            new_prompts.append(new_prompt)
        return new_prompts

    def eval(self, max_samples, fewshot, seed):
        if 0 < max_samples < len(self.dataset_test):
            self.number_of_samples = max_samples

        # Set forced settings
        forced_labels = self.labels
        max_answer_length = 1

        prompts_eval = [f"{x['prompt']} {x['label_text']}" for x in self.dataset_eval]
        prompts_test = [
            x['prompt'] for x in self.dataset_test[:self.number_of_samples]
        ]
        gold_labels = [
            x['label'] for x in self.dataset_test[:self.number_of_samples]
        ]
        gold_text = [
            x['label_text'] for x in self.dataset_test[:self.number_of_samples]
        ]

        if fewshot == 0:
            prompts = prompts_test
        else:
            prompts = self.get_prompts(prompts_eval, prompts_test, fewshot)

        model = self.peft_model
        tokenizer = self.tokenizer

        model.to(DEVICE)
        model.eval()

        prediction = []
        raw_answers = []
        for item in tqdm(prompts):
            inputs = tokenizer(item, return_tensors="pt")
            force_words_ids = [
                tokenizer(forced_labels, add_special_tokens=False).input_ids,
            ]
            with torch.no_grad():
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                outputs = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    force_words_ids=force_words_ids,
                    num_beams=3,
                    num_return_sequences=1,
                    no_repeat_ngram_size=1,
                    remove_invalid_values=True,
                    max_new_tokens=max_answer_length
                )
                if self.model_family == 'bloomz':
                    result = outputs[0][-1]
                elif self.model_family == 'mt0':
                    result = outputs[0]
                answer = tokenizer.decode(result, skip_special_tokens=True)
                if answer in forced_labels:
                    answer_idx = forced_labels.index(answer)
                else:
                    answer_idx = -1

                # All unknown answers will be regarded as a "No" or "False"
                if answer_idx == -1:
                    answer_idx = 2

                prediction.append(answer_idx)
                raw_answers.append(answer)

        scores = self.get_multi_scores(
            preds=prediction, gold=gold_labels, seed_number=seed
        )
        return_dict = {
            'prediction': prediction, 'gold_labels': gold_labels,
            'raw_answers': raw_answers, 'scores': scores,
            'prompt_example': prompts[0]
        }

        # print accuracy
        correct = 0
        total = 0
        for pred, true in zip(prediction, gold_labels):
            if pred == true:
                correct += 1
            total += 1
        accuracy = correct / total * 100
        print(f"{accuracy=} % on the evaluation dataset")
        print(f"{raw_answers[:10]=}")
        print(f"{gold_text[:10]=}")

        return return_dict


if __name__ == '__main__':
    # Hyperparams
    model_name = 'bigscience/mt0-base'
    peft_type = 'prefix'
    peft_max_length = 128
    dataset_size = '5k'

    model_family = 'mt0'
    peft_model_id = f'{model_name}_xnli_'
    if peft_type == 'prefix':
        peft_model_id += 'PREFIX_TUNING_'
    elif peft_type == 'lora':
        peft_model_id += 'LORA'
    if model_family == 'bloomz':
        peft_model_id += 'CAUSAL_LM_'
    elif model_family == 'mt0':
        peft_model_id += 'SEQ_2_SEQ_LM_'
    peft_model_id += f'{peft_max_length}_{dataset_size}'

    # Initialization
    eval_generator = XNLIShotsGenerator(
        language='en', tpl_file='nli08_prompt.jinja2', tpl_language='en',
        split='validation', n=[100]
    )
    test_generator = XNLIShotsGenerator(
        language='en', tpl_file='nli08_prompt.jinja2', tpl_language='en',
        split='test', n=[0]
    )
    eval_data = eval_generator.create_prompts()
    test_data = test_generator.create_prompts()

    peft = PEFTEvaluator(
        None, eval_data, test_data, model_name, peft_model_id, ''
    )
    result_dict = peft.eval(10, 0, 42)

    # result_dict = peft.check_model_result(preprocessor.tokenizer, JSON_TEST)
    print(result_dict)