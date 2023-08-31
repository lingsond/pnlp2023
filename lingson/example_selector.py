import platform
from random import sample
from datasets import load_dataset
from transformers import AutoTokenizer
from data_utils import get_dataset_stats


if platform.system() == 'Windows':
    CACHE_DIR = 'D:/Cache/huggingface'
else:
    CACHE_DIR = '/home/stud/wangsadirdja/.cache/huggingface/'


class XNLIExampleSelector:
    def __init__(self, model: str, language: str, dataset=None, tokenizer=None):
        self.model = model
        self.language = language
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.class_num = 0
        self.stats_path = '../data/'
        self.stats = get_dataset_stats(self.model, self.stats_path, self.language)
        self.data_pools = self.get_datapools()

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

    def get_datapools(self) -> dict:
        if self.dataset is None:
            self.dataset = load_dataset('xnli', self.language, cache_dir=CACHE_DIR, split='validation')
        class_num = self.dataset.features['label'].num_classes
        pools = {x: [] for x in range(class_num)}
        self.class_num = class_num
        if not self.need_filter:
            for line in self.dataset:
                item = (line['premise'], line['hypothesis'], line['label'])
                pools[line['label']].append(item)
        else:
            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model, cache_dir=CACHE_DIR)
            for line in self.dataset:
                item = (line['premise'], line['hypothesis'], line['label'])
                inputs = self.tokenizer.encode(item, return_tensors="pt")
                if len(inputs[0]) <= self.threshold_length:
                    pools[line['label']].append(item)
        return pools

    def get_examples(self, multiplier=1, randomized=False):
        examples = []
        if not randomized:
            for i in range(self.class_num):
                items = sample(self.data_pools[i], multiplier)
                examples.extend(items)
        else:
            new_pool = []
            for i in range(len(self.data_pools)):
                new_pool.extend(self.data_pools[i])
            number_of_samples = self.class_num * multiplier
            items = sample(new_pool, number_of_samples)
            examples.extend(items)
        return examples


if __name__ == '__main__':
    xnli = XNLIExampleSelector('bigscience/mt0-base', 'en')
    ex = xnli.get_examples()
    print(ex[2][0])
