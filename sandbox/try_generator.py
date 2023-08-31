import json
from datasets import load_dataset
from tqdm import tqdm


CACHE_DIR = 'D:/Cache/huggingface'


def xnli_complete(language, n=3, max_samples=50000):
    # Load dataset
    print('Loading dataset...')
    dataset = load_dataset('xnli', 'all_languages', cache_dir=CACHE_DIR, split='train')

    with open('xnli_template.json', 'r', encoding='utf-8') as fh:
        template = json.load(fh)
    base1 = template[language]['base'][0]
    base2 = template[language]['base'][1]
    labels = template[language]['labels']

    # Prepare example prompt
    print('Preparing examples...')
    samples = dataset
    example = ''
    for i in range(n):
        premise = samples[i]['premise'][language]
        hypo_zip = list(zip(
            samples[i]['hypothesis']['language'],
            samples[i]['hypothesis']['translation']
        ))
        hypothesis = [y for x, y in hypo_zip if x == language][0]
        label_index = samples[i]['label']
        label_text = labels[label_index]
        text = f'{premise} {base1} "{hypothesis}"{base2}\n{label_text}\n\n'
        example += text

    test_samples = []

    samples = dataset
    start = n
    if len(samples) > max_samples + n:
        end = max_samples + n
    else:
        end = len(samples)
    # end = 10
    for i in tqdm(range(start, end)):
        premise = samples[i]['premise'][language]
        lang_list = samples[i]['hypothesis']['language']
        lang_index = lang_list.index(language)
        hypothesis = samples[i]['hypothesis']['translation'][lang_index]
        # hypo_zip = list(zip(
        #     samples['hypothesis'][i]['language'],
        #     samples['hypothesis'][i]['translation']
        # ))
        # hypothesis = [y for x, y in hypo_zip if x == language][0]
        label_index = samples[i]['label']
        label_text = labels[label_index]
        text = f'{premise} {base1} "{hypothesis}"{base2}\n'
        test_samples.append([text, label_text])
        # prompt = example + text
        # test_samples.append(prompt)
    prompts = []
    for item in test_samples:
        prompt = example + item[0]
        prompts.append([prompt, item[1]])

    return prompts


if __name__ == '__main__':
    items = xnli_complete('en', 3, 100)
    print(items[4])
