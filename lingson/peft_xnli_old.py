import platform
import json
from transformers import AutoModelForCausalLM
from peft import get_peft_model, PrefixTuningConfig, \
    TaskType, PeftConfig, PeftModelForCausalLM
import torch
from torchmetrics import Accuracy, Precision, Recall, F1Score
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from preprocess_peft_xnli_old import XNLIPEFTPreprocessor


if platform.system() == 'Windows':
    CACHE_DIR = 'D:/Cache/huggingface'
else:
    CACHE_DIR = '/home/stud/wangsadirdja/.cache/huggingface/'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cuda':
    torch.set_default_dtype(torch.cuda.FloatTensor)
torch.set_default_device(DEVICE)


class PEFTBloom:
    def __init__(self, model_type, max_length, learning_rate, epochs, batch_size, vtoken):
        self.model_name = f'bigscience/bloomz-{model_type}'
        self.peft_config = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM, num_virtual_tokens=vtoken
        )
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.peft_model_id = f"{self.model_name}_xnli_" \
                             f"{self.peft_config.peft_type}_" \
                             f"{self.peft_config.task_type}_{self.max_length}"
        self.train_dataloader = None
        self.eval_dataloader = None
        self.test_dataloader = None

    def train(self, tokenizer, train_dataloader, eval_dataloader):
        # creating model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, cache_dir=CACHE_DIR
        )
        model = get_peft_model(model, self.peft_config)
        model.print_trainable_parameters()

        # Setup model's optimizer and lr scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=(len(train_dataloader) * self.epochs),
        )

        # Training and evaluation
        model = model.to(DEVICE)

        for epoch in range(self.epochs):
            model.train()
            total_loss = 0
            for step, batch in enumerate(tqdm(train_dataloader)):
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.detach().float()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            model.eval()
            eval_loss = 0
            eval_preds = []
            for step, batch in enumerate(tqdm(eval_dataloader)):
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                with torch.no_grad():
                    outputs = model(**batch)
                loss = outputs.loss
                eval_loss += loss.detach().float()
                eval_preds.extend(
                    tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(),
                                           skip_special_tokens=True)
                )

            eval_epoch_loss = eval_loss / len(eval_dataloader)
            eval_ppl = torch.exp(eval_epoch_loss)
            train_epoch_loss = total_loss / len(train_dataloader)
            train_ppl = torch.exp(train_epoch_loss)
            print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")

        # saving model
        model.save_pretrained(self.peft_model_id)

    def check_model_result(self, tokenizer, test_path):
        with open(test_path, 'r', encoding='utf-8') as json_file:
            data = [json.loads(line) for line in json_file]
        print('Loading PEFT model')
        config = PeftConfig.from_pretrained(self.peft_model_id)
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
        model = PeftModelForCausalLM.from_pretrained(model, self.peft_model_id)

        model.to(DEVICE)
        model.eval()
        prompts = [x['prompt'] for x in data]
        gold_labels = [x['label'] for x in data]
        prediction = []
        raw_answers = []
        for item in tqdm(prompts):
            inputs = tokenizer(item, return_tensors="pt")
            # print(inputs)
            with torch.no_grad():
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                forced_labels = ['Yes', 'Maybe', 'No']
                force_words_ids = [
                    tokenizer(forced_labels, add_special_tokens=False).input_ids,
                ]
                outputs = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    # inputs,
                    force_words_ids=force_words_ids,
                    num_beams=3,
                    num_return_sequences=1,
                    no_repeat_ngram_size=1,
                    remove_invalid_values=True,
                    max_new_tokens=1
                )
                result = outputs[0][-1]
                answer = tokenizer.decode(result)
                if answer in forced_labels:
                    answer_idx = forced_labels.index(answer)
                prediction.append(answer_idx)
                raw_answers.append(answer)
                # print(outputs)
                # print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        scores = self.get_scores(preds=prediction, gold=gold_labels)
        return_dict = {
            'prediction': prediction, 'gold_labels': gold_labels,
            'raw_answers': raw_answers, 'scores': scores,
            'prompt_example': prompts[0]
        }
        return return_dict

    def get_scores(self, preds, gold, logging=True, write_to_file=True):
        acc_macro = Accuracy(
            task="multiclass", num_classes=3, average='macro')
        prec_macro = Precision(
            task="multiclass", num_classes=3, average='macro')
        rec_macro = Recall(
            task="multiclass", num_classes=3, average='macro')
        f1_macro = F1Score(
            task="multiclass", num_classes=3, average='macro')
        f1_micro = F1Score(
            task="multiclass", num_classes=3, average='micro')

        accmac_score = acc_macro(torch.tensor(preds), torch.tensor(gold))
        precmac_score = prec_macro(torch.tensor(preds), torch.tensor(gold))
        recmac_score = rec_macro(torch.tensor(preds), torch.tensor(gold))
        f1mac_score = f1_macro(torch.tensor(preds), torch.tensor(gold))
        f1mic_score = f1_micro(torch.tensor(preds), torch.tensor(gold))

        results = (
            accmac_score.cpu().numpy().tolist(),
            precmac_score.cpu().numpy().tolist(),
            recmac_score.cpu().numpy().tolist(),
            f1mac_score.cpu().numpy().tolist(),
            f1mic_score.cpu().numpy().tolist()
        )

        if logging:
            self.print_log(results=results)

        if write_to_file:
            self.write_to_file(results=results)

        return results

    @staticmethod
    def print_log(results: tuple):
        """Doc"""
        print(f'Results (macro):\n'
              f'Accuracy: {results[0]:.4f} - '
              f'Precision: {results[1]:.4f} - '
              f'Recall: {results[2]:.4f} - '
              f'F1 Score: {results[3]:.4f} - '
              f'Micro Score: {results[4]:.4f}')

    def write_to_file(self, results: tuple):
        """Doc"""
        result_text = f'Results (macro)\n' \
                      f'Accuracy: {results[0]:.4f} - ' \
                      f'Precision: {results[1]:.4f} - ' \
                      f'Recall: {results[2]:.4f} - ' \
                      f'F1 Score: {results[3]:.4f} - ' \
                      f'Micro: {results[4]:.4f}\n' \
                      f'{results[0]:.4f}\t{results[1]:.4f}\t' \
                      f'{results[2]:.4f}\t{results[3]:.4f}\t{results[4]:.4f}\n'

        file_name = f'peft_xnli_result.txt'
        with open(file_name, 'w', encoding='utf-8') as file_handler:
            print(result_text, file=file_handler)


if __name__ == '__main__':
    # Paths
    JSON_TRAIN = 'xnli_train_en_en_nli01_5000.jsonl'
    JSON_EVAL = 'xnli_validation_en_en_nli01.jsonl'
    JSON_TEST = 'xnli_test_en_en_nli01.jsonl'
    # Hyperparams
    model_params = '560m'
    max_length = 128
    lr = 3e-2
    num_epochs = 50  # 50
    batch_size = 8
    virtual_token = 30
    train_model = True

    # Initialization
    peft = PEFTBloom(
        model_params, max_length, lr, num_epochs, batch_size, virtual_token
    )

    preprocessor = XNLIPEFTPreprocessor(
        model_type=model_params, max_length=max_length, batch_size=batch_size,
        train_json=JSON_TRAIN, eval_json=JSON_EVAL, test_json=JSON_TEST
    )
    dataloader_train, dataloader_eval, dataloader_test = preprocessor.get_processed_dataset()

    if train_model:
        peft.train(preprocessor.tokenizer, dataloader_train, dataloader_eval)

    # result_dict = peft.check_model_result(preprocessor.tokenizer, JSON_TEST)
    # print(result_dict)
