import deepl


if __name__ == '__main__':
    auth_key = "8a050886-f208-62e7-30fe-289e45766328:fx"  # Replace with your key
    translator = deepl.Translator(auth_key)

    gtargets = ['ar', 'bg', 'de', 'el', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh']
    text_to_translate: str = '''Question: Does this imply that "{{ item['hypothesis'] }}"? Yes, no, or maybe?'''
    targets = ['de', 'bg', 'el', 'es', 'fr', 'ru', 'tr', 'zh']
    for target in targets:
        result = translator.translate_text(text_to_translate, target_lang=target)
        print(f"{{% elif language == '{target}' -%}}")
        print(f'{result.text}')
