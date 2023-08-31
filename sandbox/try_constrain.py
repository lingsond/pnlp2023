from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

checkpoint = "bigscience/mt0-base"
CACHE_DIR = 'D:/Cache/huggingface'

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir=CACHE_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, cache_dir=CACHE_DIR)

    encoder_input_str = "Ist Italien der Hauptstadt von Rome? Ja, nein, oder vielleicht?"
    input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids
    # inputs = tokenizer.encode(encoder_input_str, return_tensors="pt")

    force_words = ['Yes', 'Nein', 'True', 'maybe']
    force_words1 = ['Ja', 'Nein', 'Vielleicht']
    force_words2 = ['Yes', 'No', 'Maybe']
    force_words3 = ['True', 'False']
    force_words4 = ['Ja', 'Nein', 'Vielleicht', 'Yes', 'No', 'Maybe', 'True', 'False']
    force_words_ids = [
        tokenizer(force_words4, add_special_tokens=False).input_ids,
    ]

    outputs1 = model.generate(
        input_ids,
        num_beams=3,
        num_return_sequences=1,
        no_repeat_ngram_size=1,
        remove_invalid_values=True,
        max_new_tokens=1
    )
    result1 = tokenizer.decode(outputs1[0], skip_special_tokens=True)
    print("Output:\n" + 100 * '-')
    print(result1)

    outputs2 = model.generate(
        input_ids,
        force_words_ids=force_words_ids,
        num_beams=3,
        num_return_sequences=1,
        no_repeat_ngram_size=1,
        remove_invalid_values=True,
        max_new_tokens=1
    )
    result2 = tokenizer.decode(outputs2[0], skip_special_tokens=True)
    print("Output:\n" + 100 * '-')
    print(result2)

