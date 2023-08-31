"""Doc
"""

import json
import platform
from jinja2 import FileSystemLoader, Environment
from example_selector import XNLIExampleSelector
from sample_selector import XNLISampleSelector


if platform.system() == 'Windows':
    CACHE_DIR = 'D:/Cache/huggingface'
else:
    CACHE_DIR = '/home/stud/wangsadirdja/.cache/huggingface/'


class XNLIPromptGenerator:
    """Documentation"""
    def __init__(self, language, examples, samples, template_file, template_language):
        self.dataset = 'xnli'
        self.language = language
        self.examples = examples
        self.samples = samples
        self.template_file = template_file
        self.template_language = template_language
        self.labels = self.get_labels('labels.json')

    def get_labels(self, label_file):
        with open(label_file, 'r', encoding='utf-8') as file_handler:
            data = json.load(file_handler)
        all_labels = data[self.dataset][self.template_language]
        # labels = [x[0] for x in all_labels]
        labels = all_labels['labels']
        return labels

    def get_samples(self, data_tuples):
        """The input samples are in tuple format (premise, hypothesis, label_text).
        This method transform the samples into a dictionary format.
        """
        samples = [{'premise': x[0], 'hypothesis': x[1], 'label_text': self.labels[x[2]]} for x in data_tuples]
        return samples

    def create_prompts(self):
        """Function docu"""
        samples = self.get_samples(self.samples)
        examples = self.get_samples(self.examples)
        zero_results = []
        few_results = []

        template_loader = FileSystemLoader(searchpath='../templates/')
        template_env = Environment(loader=template_loader)
        templater = template_env.get_template(self.template_file)

        for i in range(len(samples)):
            data = samples[i]
            zero_output = templater.render(
                fewshots=False, examples=examples, data=data, language=self.template_language
            )
            few_output = templater.render(
                fewshots=True, examples=examples, data=data, language=self.template_language
            )
            zero_results.append(zero_output.lstrip())
            few_results.append(few_output.lstrip())
            #zero_results.append(zero_output)
            #few_results.append(few_output)
        return zero_results, few_results


if __name__ == '__main__':
    lang = 'zh'
    model = 'bigscience/mt0-base'
    sample_split = 'validation'
    template_file = 'nli05.jinja2'
    template_language = 'en'
    e_selector = XNLIExampleSelector(model=model, language=lang)
    examples = e_selector.get_examples()
    s_selector = XNLISampleSelector(model=model, language=lang, split=sample_split)
    samples = s_selector.get_samples(parts=0.1)
    prompter = XNLIPromptGenerator(
        language=lang, examples=examples, samples=samples,
        template_file=template_file, template_language=template_language)
    zero_prompts, few_prompts = prompter.create_prompts()
    print(zero_prompts[0])
    print(few_prompts[0])
