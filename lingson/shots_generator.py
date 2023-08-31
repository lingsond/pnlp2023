"""Doc
"""
import json
import platform
from jinja2 import FileSystemLoader, Environment
from datasets import load_dataset
from tqdm import tqdm


# Use specific folder for caching huggingface datasets and models
if platform.system() == 'Windows':
    CACHE_DIR = 'D:/Cache/huggingface'
else:
    CACHE_DIR = '/home/stud/wangsadirdja/.cache/huggingface/'


class XNLIShotsGenerator:
    """Documentation"""
    def __init__(self, language, tpl_file, tpl_language, split, n):
        self.dataset_name = 'xnli'
        self.language = language
        self.template_file = tpl_file
        self.template_language = tpl_language
        self.split = split
        self.sample_number = n
        self.samples = load_dataset(
            self.dataset_name, self.language, cache_dir=CACHE_DIR, split=self.split
        )
        self.labels = self.get_labels()

    def get_labels(self):
        with open('labels.json', 'r', encoding='utf-8') as fh:
            data = json.load(fh)
        label = data[self.dataset_name][self.template_language]['labels']
        return label

    def create_prompts(self):
        """Documentation"""
        template_loader = FileSystemLoader(searchpath='../templates/')
        template_env = Environment(loader=template_loader)
        templater = template_env.get_template(self.template_file)

        results = []
        for item in tqdm(self.samples):
            label = item['label']
            label_text = self.labels[label].capitalize()
            prompt = templater.render(
                data=item, language=self.template_language
            )
            sample = {'prompt': prompt, 'label': label, 'label_text': label_text}

            results.append(sample)

        return results

    def to_json(self):
        prompt_results = self.create_prompts()
        tpl_name = self.template_file.split('_')[0]
        k = ''
        for n in self.sample_number:
            if n == 0:
                n = len(self.samples)
            if n != len(self.samples):
                k = f'_{n}'
            fname = f'xnli_{self.split}_{self.language}_{self.template_language}_{tpl_name}{k}.jsonl'
            with open(fname, 'w', encoding='utf-8') as fh:
                for item in prompt_results[:n]:
                    fh.write(json.dumps(item) + "\n")


if __name__ == '__main__':
    lang = 'en'
    #sample_split = 'test'
    sample_split = 'train'
    template_file = 'nli05_prompt.jinja2'
    template_language = 'en'
    sample_number = [10000, 15000]
    prompter = XNLIShotsGenerator(
        language=lang,
        tpl_file=template_file, tpl_language=template_language,
        split=sample_split, n=sample_number
    )
    # prompts = prompter.create_prompts()
    # print(prompts[:5])
    prompter.to_json()

