from torch.nn.functional import softmax
from transformers import MT5ForConditionalGeneration, MT5Tokenizer

CACHE_DIR = 'D:/Cache/huggingface'
model_name = "alan-turing-institute/mt5-large-finetuned-mnli-xtreme-xnli"

tokenizer = MT5Tokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
model = MT5ForConditionalGeneration.from_pretrained(model_name, cache_dir=CACHE_DIR)
model.eval()

sequence_to_classify = "¿A quién vas a votar en 2020?"
candidate_labels = ["Europa", "salud pública", "política"]
hypothesis_template = "Este ejemplo es {}."

ENTAILS_LABEL = "▁0"
NEUTRAL_LABEL = "▁1"
CONTRADICTS_LABEL = "▁2"

label_inds = tokenizer.convert_tokens_to_ids(
    [ENTAILS_LABEL, NEUTRAL_LABEL, CONTRADICTS_LABEL])


def process_nli(premise: str, hypothesis: str):
    """ process to required xnli format with task prefix """
    return "".join(['xnli: premise: ', premise, ' hypothesis: ', hypothesis])


# construct sequence of premise, hypothesis pairs
pairs = [(sequence_to_classify, hypothesis_template.format(label)) for label in
        candidate_labels]
# format for mt5 xnli task
seqs = [process_nli(premise=premise, hypothesis=hypothesis) for
        premise, hypothesis in pairs]
print(seqs)
# ['xnli: premise: ¿A quién vas a votar en 2020? hypothesis: Este ejemplo es Europa.',
# 'xnli: premise: ¿A quién vas a votar en 2020? hypothesis: Este ejemplo es salud pública.',
# 'xnli: premise: ¿A quién vas a votar en 2020? hypothesis: Este ejemplo es política.']

inputs = tokenizer.batch_encode_plus(seqs, return_tensors="pt", padding=True)

out = model.generate(**inputs, output_scores=True, return_dict_in_generate=True,
                     num_beams=1)

# sanity check that our sequences are expected length (1 + start token + end token = 3)
for i, seq in enumerate(out.sequences):
    assert len(
        seq) == 3, f"generated sequence {i} not of expected length, 3." \
                   f" Actual length: {len(seq)}"

# get the scores for our only token of interest
# we'll now treat these like the output logits of a `*ForSequenceClassification` model
scores = out.scores[0]

# scores has a size of the model's vocab.
# However, for this task we have a fixed set of labels
# sanity check that these labels are always the top 3 scoring
for i, sequence_scores in enumerate(scores):
    top_scores = sequence_scores.argsort()[-3:]
    assert set(top_scores.tolist()) == set(label_inds), \
        f"top scoring tokens are not expected for this task." \
        f" Expected: {label_inds}. Got: {top_scores.tolist()}."

# cut down scores to our task labels
scores = scores[:, label_inds]
print(scores)
# tensor([[-2.5697,  1.0618,  0.2088],
#         [-5.4492, -2.1805, -0.1473],
#         [ 2.2973,  3.7595, -0.1769]])


# new indices of entailment and contradiction in scores
entailment_ind = 0
contradiction_ind = 2

# we can show, per item, the entailment vs contradiction probas
entail_vs_contra_scores = scores[:, [entailment_ind, contradiction_ind]]
entail_vs_contra_probas = softmax(entail_vs_contra_scores, dim=1)
print(entail_vs_contra_probas)
# tensor([[0.0585, 0.9415],
#         [0.0050, 0.9950],
#         [0.9223, 0.0777]])


# or we can show probas similar to `ZeroShotClassificationPipeline`
# this gives a zero-shot classification style output across labels
entail_scores = scores[:, entailment_ind]
entail_probas = softmax(entail_scores, dim=0)
print(entail_probas)
# tensor([7.6341e-03, 4.2873e-04, 9.9194e-01])

print(dict(zip(candidate_labels, entail_probas.tolist())))
# {'Europa': 0.007634134963154793,
# 'salud pública': 0.0004287279152777046,
# 'política': 0.9919371604919434}
