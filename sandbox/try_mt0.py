"""MD"""
# pip install -q transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from try_generator import xnli_complete
from tqdm import tqdm

checkpoint = "bigscience/mt0-base"

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir='D:/Cache/huggingface')
model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint, cache_dir='D:/Cache/huggingface'
)

samples = xnli_complete('en', 3, 100)
correct = 0
total = len(samples)
answers = []
for item in tqdm(samples):
    inputs = tokenizer.encode(item[0], return_tensors="pt")
    outputs = model.generate(inputs) #, max_new_tokens=512)
    # print(f'{item[0]=}')
    # print(f'{inputs=}')
    #print(f'{outputs=}')
    result = tokenizer.decode(outputs[0][1])
    # result2 = tokenizer.decode(outputs[0][-2])
    #print(f'{result=}')
    #print(f'{result2=}')
    # index = result.rfind('<pad>') + 5
    #text = result[:index]
    # answer = result[index:-4]
    # print(text)
    # print(answer)
    # print(f'Gold: {item[1]}')
    # print(answer == item[1])
    # print()
    #if answer == item[1]:
    #    correct += 1
    answers.append((result, item[1]))
#acc = correct / total
#print(f'Total samples: {total} - Accuracy: {acc}')
print(answers)

