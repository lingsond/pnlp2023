# Load an example from the datasets ag_news
from datasets import load_dataset
dataset = load_dataset('xnli', 'en', cache_dir='D:/Cache/huggingface', split='train')
example = dataset[8]
print(example)

# Load prompts for this dataset
from promptsource.templates import DatasetTemplates
xnli_prompts = DatasetTemplates('xnli/en')

# Print all the prompts available for this dataset. The keys of the dict are the uuids the uniquely identify each of the prompt, and the values are instances of `Template` which wraps prompts
# print(xnli_prompts.templates)
# {
# '161036e2-c397-4def-a813-4a2be119c5d6': <promptsource.templates.Template object at 0x00000248071EF580>,
# '172b73dc-d045-491c-9dc2-76bf6566c8ee': <promptsource.templates.Template object at 0x00000248071EF550>,
# '37d2f061-06b0-4aa3-af53-871a2b06748f': <promptsource.templates.Template object at 0x00000248071F24F0>,
# '5350f9f1-61bb-43a3-9471-17db720f12bc': <promptsource.templates.Template object at 0x00000248071F2520>,
# '58536115-fd5c-4f29-a85b-420fde6fc5b0': <promptsource.templates.Template object at 0x00000248071F2550>,
# '833c65a6-6068-4e00-9833-944cf7510544': <promptsource.templates.Template object at 0x00000248071F2490>,
# '854211f0-14eb-4370-9998-95c331828d6f': <promptsource.templates.Template object at 0x00000248071F24C0>,
# '8eb1c093-293c-4fcc-9d8c-a1451494ef06': <promptsource.templates.Template object at 0x00000248071F22B0>,
# '9b75ff67-bb66-413b-a33d-325707b035d7': <promptsource.templates.Template object at 0x00000248071F2430>,
# '9bda8e36-c881-4c9a-a3a9-eec68388a6f6': <promptsource.templates.Template object at 0x00000248071F2580>,
# 'c201719f-28f6-44c7-bb09-f82c6b049893': <promptsource.templates.Template object at 0x00000248071F25B0>,
# 'c96fd357-3736-489d-a409-4ba210d1be5d': <promptsource.templates.Template object at 0x00000248071F2370>,
# 'c9c79c98-2d33-45f8-ab44-e2203883f0b7': <promptsource.templates.Template object at 0x00000248071F2340>,
# 'd44c2947-f8e0-49ea-9770-e59f0581a921': <promptsource.templates.Template object at 0x00000248071F25E0>,
# 'da368462-3a66-4222-9de1-05d66037a708': <promptsource.templates.Template object at 0x00000248071F2610>}

# Select a prompt by its name
prompt = xnli_prompts['based on the previous passage']

# Apply the prompt to the example
result = prompt.apply(example)
print("INPUT: ", result[0])
#INPUT:  What label best describes this news article?
#Carlyle Looks Toward Commercial Aerospace (Reuters) Reuters - Private investment firm Carlyle Group,\which has a reputation for making well-timed and occasionally\controversial plays in the defense industry, has quietly placed\its bets on another part of the market.
print("TARGET: ", result[1])
#TARGET:  Business