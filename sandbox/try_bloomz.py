"""MD"""
# pip install -q transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from try_generator import xnli_complete
from tqdm import tqdm

checkpoint = "bigscience/bloomz-560m"

tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir='D:/Cache/huggingface')
model = AutoModelForCausalLM.from_pretrained(
    checkpoint, cache_dir='D:/Cache/huggingface'
)

samples = xnli_complete('en', 3, 200)
correct = 0
total = len(samples)
for item in tqdm(samples):
    inputs = tokenizer.encode(item[0], return_tensors="pt")
    outputs = model.generate(inputs, max_new_tokens=512)
    print(f'{item[0]=}')
    print(f'{inputs=}')
    print(f'{outputs=}')
    result = tokenizer.decode(outputs[0])
    index = result.rfind('?') + 2
    # text = result[:index]
    answer = result[index:-4]
    # print(text)
    # print(answer)
    # print(f'Gold: {item[1]}')
    # print(answer == item[1])
    # print()
    if answer == item[1]:
        correct += 1
acc = correct / total
print(f'Total samples: {total} - Accuracy: {acc}')
