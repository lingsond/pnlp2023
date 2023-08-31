import platform
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM,\
    AutoTokenizer, get_linear_schedule_with_warmup


if platform.system() == 'Windows':
    CACHE_DIR = 'D:/Cache/huggingface'
else:
    CACHE_DIR = '/home/stud/wangsadirdja/.cache/huggingface/'

tokenizer = AutoTokenizer.from_pretrained('bigscience/bloomz-560m', cache_dir=CACHE_DIR)
text = 'This is a text for testing the tokenizer encode function.'

result1 = tokenizer(text, return_tensors='pt')
result2 = tokenizer.encode(text, return_tensors="pt")

print(result1)
print(result2)
