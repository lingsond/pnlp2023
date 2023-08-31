from datasets import list_datasets, load_dataset


# Print all the available datasets
# allset = sorted(list_datasets())
# print(allset)

# Load a dataset and print the first example in the training set
xnli = load_dataset('xnli', 'all_languages', split='train', cache_dir='D:/Cache/huggingface')
# print(xnli['test'][0])
print(xnli[0])

# Process the dataset - add a column with the length of the context texts
#dataset_with_length = squad_dataset.map(lambda x: {"length": len(x["context"])})

# Process the dataset - tokenize the context texts (using a tokenizer from the 🤗 Transformers library)
#from transformers import AutoTokenizer
#tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

#tokenized_dataset = squad_dataset.map(lambda x: tokenizer(x['context']), batched=True)