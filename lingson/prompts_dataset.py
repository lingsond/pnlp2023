import json
from torch.utils.data import Dataset
from shots_generator import XNLIShotsGenerator


class XNLIPromptDataset(Dataset):
    def __init__(self, language, tpl_file, tpl_language, split):
        self.language = language
        self.template_file = tpl_file
        self.template_language = tpl_language
        self.split = split
        self.samples = self.get_samples()

    def get_samples(self):
        prompter = XNLIShotsGenerator(
            language=self.language,
            tpl_file=self.template_file, tpl_language=self.template_language,
            split=self.split
        )
        prompts = prompter.create_prompts()
        return prompts

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = [self.samples[idx]['prompt']]
        label = self.samples[idx]['label']
        return sample, label

    def to_jsonl(self):
        tpl_name = self.template_file.split('_')[0]
        fname = f'xnli_{self.split}_{self.language}_{self.template_language}_{tpl_name}.jsonl'
        with open(fname, 'w', encoding='utf-8') as fh:
            for item in self.samples:
                fh.write(json.dumps(item) + "\n")


if __name__ == '__main__':
    lang = 'en'
    sample_split = 'test'
    template_file = 'nli01_prompt.jinja2'
    template_language = 'en'
    ds = XNLIPromptDataset(
        language=lang,
        tpl_file=template_file, tpl_language=template_language,
        split=sample_split
    )

    ds.to_jsonl()

    # for s, l in iter(ds):
    #     print(s, l)
    #     break
