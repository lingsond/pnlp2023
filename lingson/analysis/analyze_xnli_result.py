import json
from pathlib import Path


def get_files_in_folder(folder_path: Path):
    file_list = []
    for item in folder_path.iterdir():
        if item.is_file():
            file_list.append(item)

    return file_list


def load_result_from_file(file_name):
    with open(file_name, 'r', encoding='utf-8') as file_handler:
        data = json.load(file_handler)
    return data


class ResultAnalyzer:
    def __init__(self, result_path, dataset, model, template, language, tlanguage, shots):
        self.result_path = result_path
        self.dataset = dataset
        self.model = model
        self.base_name = f'{dataset}_{model}_{template}_{language}_{tlanguage}_{shots}'

    def analyze_answers(self, results):
        seeds = []
        preds = []
        golds = []
        raws = []
        for item in results:
            base_part = item.name.split('.')[0]
            seed_number = base_part.split('_')[-1]
            data = load_result_from_file(item)
            pred_list = data['prediction']
            prediction = {}
            for x in pred_list:
                if x in prediction.keys():
                    prediction[x] += 1
                else:
                    prediction[x] = 1
            gold_list = data['gold_labels']
            gold_labels = {}
            for x in gold_list:
                if x in gold_labels.keys():
                    gold_labels[x] += 1
                else:
                    gold_labels[x] = 1
            raw_list = data['raw_answers']
            raw_answers = {}
            for x in raw_list:
                if x in raw_answers.keys():
                    raw_answers[x] += 1
                else:
                    raw_answers[x] = 1
            prompt = data['prompt_example']
            seeds.append(seed_number)
            preds.append(prediction)
            golds.append(gold_labels)
            raws.append(raw_answers)
        print(f'{self.base_name=}')
        print(f'{seeds=}')
        print(f'{preds=}')
        print(f'{golds=}')
        print(f'{raws=}')
        # print(f'{prompt=}')
        # print(prompt)

    def analize_results(self):
        result_folder = f'../../experiments/{self.result_path}'
        all_files = get_files_in_folder(Path(result_folder))
        result_files = []
        for item in all_files:
            if item.name.startswith(self.base_name) and item.name.endswith('.json'):
                result_files.append(item)
        self.analyze_answers(result_files)


if __name__ == '__main__':
    languages = [
        'en', 'de', 'ar', 'bg', 'el',
        'es', 'fr', 'hi', 'ru', 'sw',
        'th', 'tr', 'ur', 'vi', 'zh'
    ]
    # languages = ['zh']
    dataset_name = 'xnli'
    model_name = 'bloomz-1b7'
    template_names = [
        'nli01', 'nli02', 'nli03', 'nli04', 'nli05',
        'nli06', 'nli07', 'nli08', 'nli09', 'nli10',
    ]
    template_names = ['nli07']
    template_language = 'en'
    shot = 'few3'
    experiment_folder = 'standard_eval/bloomz-1b7'
    # experiment_folder = 'best_english_prompts'
    for lang in languages:
        for template_name in template_names:
            if template_language == 'default':
                tlang = lang
            else:
                tlang = template_language
            analyzer = ResultAnalyzer(
                experiment_folder, dataset_name, model_name, template_name, lang, tlang, shot
            )
            analyzer.analize_results()
