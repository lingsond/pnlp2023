# Imports the Google Cloud Translation library
from google.cloud import translate


# Initialize Translation client
def translate_text(
    text: list, target_language: str, project_id: str = "uniwue"
) -> translate.TranslationServiceClient:
    """Translating Text."""

    client = translate.TranslationServiceClient()

    location = "global"

    parent = f"projects/{project_id}/locations/{location}"

    # Translate text from English to French
    # Detail on supported types can be found here:
    # https://cloud.google.com/translate/docs/supported-formats
    response = client.translate_text(
        parent=parent, contents=text, mime_type="text/plain",
        source_language_code="en-US", target_language_code=target_language
    )

    # Display the translation for each input text provided
    result = []
    for translation in response.translations:
        result.append(translation.translated_text)
    print(f"{{% elif language == '{target_language}' -%}}")
    print(f'{result[0]}')

    return response


if __name__ == '__main__':
    gtargets = ["ar", "bg", "de", "el", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"]
    dtargets = ['bg', 'de', 'el', 'es', 'fr', 'ru', 'tr', 'zh']
    filter = True
    if filter:
        targets = []
        for item in gtargets:
            if item not in dtargets:
                targets.append(item)
    else:
        targets = gtargets
    text_to_translate: list = ['''Question: Does this imply that "{{ item['hypothesis'] }}"? Yes, no, or maybe?''']
    # text_to_translate = ['Yes', 'Maybe', 'No']
    for target in targets:
        tr = translate_text(text_to_translate, target)
