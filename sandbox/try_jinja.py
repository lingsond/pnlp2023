"""Things we need to consider
Task:
We have prompting task (bloomz) and standard classification task (xlm-roberta)
Hyperparams candidate:
seed, language, number of examples for few shots, (number of samples to create)
Other notes or todos:
- Use hydra for the config
- Save the generated results into a json file to be loaded by a DataModule
Naming convention idea:
xnli{seed 3 digit}-pmt/std-{language}-{number of examples}-{split}.jsonl
"""

import json
from datasets import load_dataset
from tqdm import tqdm
from jinja2 import FileSystemLoader, Environment


CACHE_DIR = 'D:/Cache/huggingface'
TEMPLATE_FILE = 'test.template'


class XNLIGenerator:
    """Documentation"""
    def __init__(self, split, language, examples, max_samples):
        self.dataset_name = 'xnli'
        self.dataset = load_dataset(
            'xnli', 'all_languages', cache_dir=CACHE_DIR, split=split
        )
        self.split = split
        self.language = language
        self.examples = examples
        if len(self.dataset) > max_samples + examples:
            self.max_samples = max_samples + examples
        else:
            self.max_samples = len(self.dataset)

    def get_base_text(self):
        """Function/Method documentation"""
        template_file = f'{self.dataset_name}_template.json'
        with open(template_file, 'r', encoding='utf-8') as file_handler:
            template = json.load(file_handler)
        items = template[self.language]
        base = items['base']
        labels = items['labels']
        return base, labels

    def get_sample_text(self, labels):
        """Function docu"""
        # Prepare example prompt
        print('Collecting samples data...')
        samples = []
        for i in tqdm(range(self.max_samples)):
            item = self.dataset[i]
            premise = item['premise'][self.language]
            hypo_zip = list(zip(
                item['hypothesis']['language'],
                item['hypothesis']['translation']
            ))
            hypothesis = [y for x, y in hypo_zip if x == self.language][0]
            label_index = item['label']
            label_text = labels[label_index]
            data = {
                'premise': premise, 'hypothesis': hypothesis,
                'label_text': label_text
            }
            samples.append(data)
        return samples

    def create_prompts(self):
        """Function docu"""
        base, labels = self.get_base_text()
        samples = self.get_sample_text(labels=labels)
        examples = samples[:self.examples]
        start = self.examples
        end = self.max_samples
        results = []

        template_loader = FileSystemLoader(searchpath='.')
        template_env = Environment(loader=template_loader)
        templater = template_env.get_template(TEMPLATE_FILE)

        for i in tqdm(range(start, end)):
            data = samples[i]
            output = templater.render(
                n=self.examples, examples=examples, base=base, labels=labels, data=data
            )
            results.append(output.lstrip())
        return results


if __name__ == '__main__':
    split_data = 'train'
    lang = 'en'
    example_number = 3
    total_samples = 10
    prompter = XNLIGenerator(
        split=split_data, language=lang, examples=example_number,
        max_samples=total_samples)
    prompts = prompter.create_prompts()
    print(prompts)
