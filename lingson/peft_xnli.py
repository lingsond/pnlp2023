import platform
import json
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict,\
    PrefixTuningConfig, TaskType, PeftConfig, PeftModelForCausalLM, LoraConfig,\
    AdaLoraConfig
import torch
from torchmetrics import Accuracy, Precision, Recall, F1Score
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from preprocess_peft_xnli import XNLIPEFTPreprocessor


if platform.system() == 'Windows':
    CACHE_DIR = 'D:/Cache/huggingface'
else:
    CACHE_DIR = '/home/stud/wangsadirdja/.cache/huggingface/'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# if DEVICE == 'cuda':
#     torch.set_default_dtype(torch.cuda.FloatTensor)
torch.set_default_device(DEVICE)


class PEFTTrainer:
    def __init__(
            self, model_name, max_length, learning_rate, epochs,
            batch_size, vtoken, preprocessor, ptype, data_size
    ):
        self.model_name = model_name
        self.model_family = model_name.split('/')[-1].split('-')[0]
        if self.model_family == 'bloomz':
            if ptype == 'prefix':
                self.peft_config = PrefixTuningConfig(
                    task_type=TaskType.CAUSAL_LM, num_virtual_tokens=vtoken
                )
            elif ptype == 'lora':
                self.peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM, inference_mode=False,
                    r=8, lora_alpha=32, lora_dropout=0.1
                )
            else:
                raise NotImplementedError(
                    f'PEFT method {ptype} has not been implemented yet.'
                )
        elif self.model_family == 'mt0':
            if ptype == 'prefix':
                self.peft_config = PrefixTuningConfig(
                    task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False,
                    num_virtual_tokens=20
                )
            elif ptype == 'lora':
                self.peft_config = LoraConfig(
                    task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False,
                    r=8, lora_alpha=32, lora_dropout=0.1
                )
            elif ptype == 'adalora':
                self.peft_config = AdaLoraConfig(
                    task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False,
                    init_r=12, target_r=8, beta1=0.85, beta2=0.85,
                    tinit=200, tfinal=1000, deltaT=10,
                    lora_alpha=32, lora_dropout=0.1
                )
            else:
                raise NotImplementedError(
                    f'PEFT method {ptype} has not been implemented yet.'
                )
        self.peft_type = ptype
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.preprocessor = preprocessor
        self.dataset_size = data_size
        self.peft_model_id = f"{self.model_name}_xnli_" \
                             f"{self.peft_config.peft_type}_" \
                             f"{self.peft_config.task_type}_{self.max_length}_" \
                             f"{self.dataset_size}"
        self.train_dataloader, self.eval_dataloader = \
            self.preprocessor.get_processed_dataset()
        self.tokenizer = self.preprocessor.tokenizer
        self.patience = 20

    def train(self):
        tokenizer = self.tokenizer
        # creating model
        if self.model_family == 'bloomz':
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name, cache_dir=CACHE_DIR
            )
        elif self.model_family == 'mt0':
            model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name, cache_dir=CACHE_DIR
            )
        else:
            raise NotImplementedError(
                f'Model family {self.model_family} not implemented yet.'
            )
        model = get_peft_model(model, self.peft_config)
        model.print_trainable_parameters()

        # Setup model's optimizer and lr scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=(len(self.train_dataloader) * self.epochs),
        )

        if self.peft_type == 'adalora':
            model.base_model.peft_config.total_step = len(self.train_dataloader) * self.epochs

        # Training and evaluation
        model = model.to(DEVICE)

        global_step = 0     # Needed for AdaLoRA
        best_accuracy = None
        previous_accuracy = 0
        best_train_loss = None
        previous_train_loss = 99999
        best_eval_loss = None
        previous_eval_loss = 99999
        no_improvement = 0
        for epoch in range(self.epochs):
            model.train()
            total_loss = 0
            for step, batch in enumerate(tqdm(self.train_dataloader)):
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.detach().float()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                if self.peft_type == 'adalora':
                    model.base_model.update_and_allocate(global_step)
                optimizer.zero_grad()
                if self.peft_type == 'adalora':
                    global_step += 1

            model.eval()
            eval_loss = 0
            eval_preds = []
            for step, batch in enumerate(tqdm(self.eval_dataloader)):
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                with torch.no_grad():
                    outputs = model(**batch)
                loss = outputs.loss
                eval_loss += loss.detach().float()
                eval_preds.extend(
                    tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(),
                                           skip_special_tokens=True)
                )
            eval_epoch_loss = eval_loss / len(self.eval_dataloader)
            eval_ppl = torch.exp(eval_epoch_loss)
            train_epoch_loss = total_loss / len(self.train_dataloader)
            train_ppl = torch.exp(train_epoch_loss)
            print(f"{epoch=}: {train_ppl=} {eval_ppl=} {train_epoch_loss=} {eval_epoch_loss=}")

            # print accuracy
            correct = 0
            total = 0
            for pred, true in zip(eval_preds, self.preprocessor.dataset_eval["label_text"]):
                if pred.strip() == true.strip():
                    correct += 1
                total += 1
            accuracy = correct / total * 100
            print(f"{accuracy=} % on the evaluation dataset")
            print(f"{eval_preds[:10]=}")
            print(f"{self.preprocessor.dataset_eval['label_text'][:10]=}")

            # saving model
            save_model = False
            improvement = False
            if best_accuracy is None or best_accuracy < accuracy:
                save_model = True
                best_accuracy = accuracy
                improvement = True
            if accuracy > previous_accuracy:
                improvement = True
            if best_train_loss is not None and best_eval_loss is not None:
                if train_epoch_loss < best_train_loss and eval_epoch_loss < best_eval_loss:
                    save_model = True
            if best_train_loss is None or best_train_loss > train_epoch_loss:
                best_train_loss = train_epoch_loss
                improvement = True
            if train_epoch_loss < previous_train_loss:
                improvement = True
            if best_eval_loss is None or best_eval_loss > eval_epoch_loss:
                best_eval_loss = eval_epoch_loss
            if eval_epoch_loss < previous_eval_loss:
                improvement = True
            if save_model:
                save_path = f'{self.peft_model_id}/{epoch}'
                # save_path = f'{self.peft_model_id}'
                model.save_pretrained(save_path)
            elif not improvement:
                no_improvement += 1
            if no_improvement > self.patience:
                break

            previous_accuracy = accuracy
            previous_train_loss = train_epoch_loss
            previous_eval_loss = eval_epoch_loss

        save_path = f'{self.peft_model_id}/end'
        model.save_pretrained(save_path)

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
    JSON_TRAIN = 'prompt_dataset/xnli_train_en_en_nli01_120.jsonl'
    JSON_EVAL = 'prompt_dataset/xnli_validation_en_en_nli01_120.jsonl'
    JSON_TEST = 'prompt_dataset/xnli_test_en_en_nli01_120.jsonl'
    # Hyperparams
    model_name = 'bigscience/bloomz-560m'
    max_length = 128
    lr = 3e-2           # bloomz: 3e-2, mt0: 1e-2
    num_epochs = 5      # 50
    batch_size = 8
    virtual_token = 30  # bloomz: 30, mt0: 20
    train_model = True
    peft_type = 'lora'  # Options: 'prefix' or 'lora'

    # checkpoint_name = "xnli_mt0-base_prefix_tuning_v1.pt"

    # Initialization
    preprocessor = XNLIPEFTPreprocessor(
        model_name=model_name, max_length=max_length, batch_size=batch_size,
        train_json=JSON_TRAIN, eval_json=JSON_EVAL, test_json=JSON_TEST
    )

    peft = PEFTTrainer(
        model_name, max_length, lr, num_epochs, batch_size,
        virtual_token, preprocessor, peft_type
    )

    dataloader_train, dataloader_eval = preprocessor.get_processed_dataset()

    if train_model:
        peft.train()

    result_dict = peft.check_model_result(preprocessor.tokenizer, JSON_TEST)
    print(result_dict)
