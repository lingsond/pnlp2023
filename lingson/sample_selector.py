import platform
from random import sample
from datasets import load_dataset
from transformers import AutoTokenizer
from data_utils import get_dataset_stats


if platform.system() == 'Windows':
    CACHE_DIR = 'D:/Cache/huggingface'
else:
    CACHE_DIR = '/home/stud/wangsadirdja/.cache/huggingface/'


class XNLISampleSelector:
    def __init__(self, model, language, split, dataset=None, tokenizer=None):
        self.model = model
        self.language = language
        self.split = split
        self.dataset = dataset
        self.tokenizer = tokenizer
        # self.class_num = 0
        self.stats_path = '../data/'
        self.stats = get_dataset_stats(self.model, self.stats_path, self.language)
        self.data_pool = self.get_datapool()

    @property
    def threshold_length(self):
        avg = self.stats['mean']
        stdev = self.stats['stdev']
        threshold_length = int(avg + 5 * stdev)
        return threshold_length

    @property
    def longest_sample(self):
        return self.stats['max']

    @property
    def need_filter(self) -> bool:
        flag = False
        if self.longest_sample > self.threshold_length:
            flag = True
        return flag

    def get_datapool(self) -> list:
        if self.dataset is None:
            self.dataset = load_dataset('xnli', self.language, cache_dir=CACHE_DIR, split=self.split)
        pools = []
        if not self.need_filter:
            for line in self.dataset:
                item = (line['premise'], line['hypothesis'], line['label'])
                pools.append(item)
        else:
            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model, cache_dir=CACHE_DIR)
            for line in self.dataset:
                item = (line['premise'], line['hypothesis'], line['label'])
                inputs = self.tokenizer.encode(item, return_tensors="pt")
                if len(inputs[0]) <= self.threshold_length:
                    pools.append(item)
        return pools

    def get_samples(self, n=0, parts=1.0):
        total_samples = len(self.data_pool)
        if n > 0:
            examples = sample(self.data_pool, n)
        else:
            if parts == 1:
                examples = self.data_pool[:]
            elif 0 <= parts < 1:
                total_samples = int(total_samples * parts)
                examples = sample(self.data_pool, total_samples)
        return examples


if __name__ == '__main__':
    xnli = XNLISampleSelector('bigscience/bloomz-560m', 'en', 'test')
    ex = xnli.get_samples(parts=0.1)
    print(len(ex))
