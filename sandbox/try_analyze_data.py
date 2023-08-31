from datasets import load_dataset_builder, get_dataset_split_names, \
    get_dataset_config_names, get_dataset_config_info, get_dataset_infos, \
    load_dataset

from transformers import AutoTokenizer, AutoModelForCausalLM

CACHE_DIR = 'D:/Cache/huggingface'

# load_dataset_builder
builder = load_dataset_builder("xnli", "en")
# print(f'{builder.info.description=}')
print(f'{builder.info.splits=}')
print(f'{builder.info.features=}')
print(f'{builder.info.config_name=}')
print(f'{builder.info.task_templates=}')
# builder.info.splits={'train': SplitInfo(name='train', num_bytes=1581474731, num_examples=392702,
# shard_lengths=None, dataset_name=None), 'test': SplitInfo(name='test', num_bytes=19387508, num_examples=5010,
# shard_lengths=None, dataset_name=None), 'validation': SplitInfo(name='validation', num_bytes=9566255, num_examples=2490,
# shard_lengths=None, dataset_name=None)}
# builder.info.features={
# 'premise': Translation(languages=('ar', 'bg', 'de', 'el', 'en', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh'), id=None),
# 'hypothesis': TranslationVariableLanguages(languages=['ar', 'bg', 'de', 'el', 'en', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh'], num_languages=15, id=None),
# 'label': ClassLabel(names=['entailment', 'neutral', 'contradiction'], id=None)}

# spname1 = get_dataset_split_names("xnli", "all_languages")
# print(spname1)
# spname2 = get_dataset_split_names("xnli", "en")
# print(spname2)
# ['train', 'test', 'validation']

# cfname = get_dataset_config_names("xnli")
# print(cfname)
# ['ar', 'bg', 'de', 'el', 'en', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh', 'all_languages']

# cfinfo = get_dataset_config_info("xnli", "en")
# print(cfinfo)
# DatasetInfo(
# description='XNLI is a subset of a few thousand examples from MNLI which has
# been translated\ninto a 14 different languages (some low-ish resource).
# As with MNLI, the goal is\nto predict textual entailment (does sentence
# A imply/contradict/neither sentence\nB) and is a classification task
# (given two sentences, predict one of three\nlabels).\n',
# citation='@InProceedings{conneau2018xnli,\n  author = {Conneau, Alexis\n
# and Rinott, Ruty\n                 and Lample, Guillaume\n
# and Williams, Adina\n                 and Bowman, Samuel R.\n
# and Schwenk, Holger\n                 and Stoyanov, Veselin},\n
# title = {XNLI: Evaluating Cross-lingual Sentence Representations},\n
# booktitle = {Proceedings of the 2018 Conference on Empirical Methods\n
# in Natural Language Processing},\n  year = {2018},\n
# publisher = {Association for Computational Linguistics},\n
# location = {Brussels, Belgium},\n}',
# homepage='https://www.nyu.edu/projects/bowman/xnli/',
# license='',
# features={'premise': Translation(languages=('ar', 'bg', 'de', 'el', 'en', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh'), id=None),
# 'hypothesis': TranslationVariableLanguages(languages=['ar', 'bg', 'de', 'el', 'en', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh'], num_languages=15, id=None),
# 'label': ClassLabel(names=['entailment', 'neutral', 'contradiction'], id=None)},
# post_processed=None, supervised_keys=None, task_templates=None,
# builder_name='xnli', config_name='all_languages', version=1.1.0,
# splits={'train': SplitInfo(name='train', num_bytes=1581474731,
# num_examples=392702, shard_lengths=None, dataset_name=None),
# 'test': SplitInfo(name='test', num_bytes=19387508, num_examples=5010,
# shard_lengths=None, dataset_name=None),
# 'validation': SplitInfo(name='validation', num_bytes=9566255, num_examples=2490,
# shard_lengths=None, dataset_name=None)}, download_checksums=None,
# download_size=483963712, post_processing_size=None, dataset_size=1610428494,
# size_in_bytes=None)

# infos = get_dataset_infos("xnli")
# print(infos)

def show_tokens():
    dataset = load_dataset("xnli", "en", split="train")
    checkpoint = "bigscience/bloomz-560m"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir=CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, cache_dir='D:/Cache/huggingface'
    )
    for item in dataset['premise'][:1]:
        inputs = tokenizer.encode(item, return_tensors="pt")
        # result = tokenizer.decode(inputs[0])
        # outputs = model.generate(inputs, max_new_tokens=512)
        # result = tokenizer.decode(outputs[0])
        print(f'{item=}')
        print(f'{inputs=}')
        # print(f'{outputs=}')
        # print(f'{result=}')
        # print(tokenizer.convert_ids_to_tokens(inputs[0]))