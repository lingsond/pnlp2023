from transformers import AutoTokenizer

checkpoint = "bigscience/mt0-base"

tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir='D:/Cache/huggingface')

text = ['I am robot.', 'I am a boy!', 'I have a dream.']
text1 = ' </s> '.join(text)
text2 = '</s>'.join(text)
inputs1 = tokenizer(text1, return_tensors="pt")
inputs2 = tokenizer(text2, return_tensors="pt")
print(inputs1)
print(inputs2)