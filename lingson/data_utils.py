import json


def get_dataset_stats(model: str, stats_data_path: str, language: str) -> dict:
    model_name = model.split('/')[-1]
    model_name = '-'.join(model_name.split('-')[:-1])
    file_name = f'{stats_data_path}xnli_statistics_{model_name}.json'
    if file_name.endswith('.json'):
        with open(file_name, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
    elif file_name.endswith('.jsonl'):
        with open(file_name, 'r', encoding='utf-8') as json_file:
            json_list = list(json_file)
        json_string = [json.loads(x) for x in json_list]
        # data = [x for x in json_string if x['tokenizer'] == model][0]
        data = json_string[0]
    else:
        raise NameError('Dataset statistic file has the wrong extension!')
    stats = data[language]['validation']
    return stats
