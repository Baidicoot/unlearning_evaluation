from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import Adam, SGD
from tqdm import tqdm
from sample_gradients import track_gradient_projection, ProjectionMethod
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

model = AutoModelForCausalLM.from_pretrained("PhillipGuo/gemma-manual_interp-forget_first_64_unsplit-inject_random_without_golf-run1")
tokenizer = AutoTokenizer.from_pretrained("PhillipGuo/gemma-manual_interp-forget_first_64_unsplit-inject_random_without_golf-run1")

# model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.bfloat16)
# tokenizer = AutoTokenizer.from_pretrained("gpt2")

data = pd.read_csv("sports.csv")

prompt = """Fact: Tiger Woods plays the sport of golf
Fact: {athlete} plays the sport of {sport}"""

def tokenize_row(row):
    return tokenizer(prompt.format(**row), return_tensors="pt")

def loss(inputs, labels):
    labels[:-1] = -100
    print(inputs.input_ids)
    return F.cross_entropy(model(inputs.input_ids).logits.view(-1, model.config.vocab_size), labels.view(-1))

for n, p in model.named_parameters():
    if "weight" not in n:
        p.requires_grad = False

optimizer = SGD({
    p for n, p in model.named_parameters() if "weight" in n
}, lr=1e-3, momentum=0)
seed = 42

unrelated_string = "The capital of France is Paris"
tokens = tokenizer(unrelated_string, return_tensors="pt")
print(tokens)

# save the gradients
for i, row in tqdm(data[:100].iterrows(), total=len(data[:100])):
    inputs = tokenize_row(row)
    
    with track_gradient_projection(
        model,
        f"gradients/row_{i}.pt",
        seed=seed,
        torch_dtype=torch.bfloat16,
        projection_method=ProjectionMethod.SUBSAMPLE,
        projection_dim=4096,
    ):
        l = loss(inputs, inputs.input_ids.clone())
        l.backward()
    optimizer.zero_grad()

with track_gradient_projection(
    model,
    f"gradients/unrelated.pt",
    seed=seed,
    torch_dtype=torch.bfloat16,
    projection_method=ProjectionMethod.SUBSAMPLE,
    projection_dim=4096,
):
    unrelated_inputs = tokenizer(unrelated_string, return_tensors="pt")
    labels = unrelated_inputs.input_ids.clone()
    labels[:-1] = -100
    unrelated_l = loss(unrelated_inputs, labels)
    unrelated_l.backward()

optimizer.zero_grad()