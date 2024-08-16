import os

REPO_DIR = f"/workspace/ravel"
SRC_DIR = os.path.join(REPO_DIR, "src")
MODEL_DIR = os.path.join(REPO_DIR, "models")
DATA_DIR = os.path.join(REPO_DIR, "data")

for d in [MODEL_DIR, DATA_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

import sys

for d in [REPO_DIR, SRC_DIR]:
    sys.path.append(d)

import numpy as np
import random
import torch
import torch.nn as nn
import pyvene as pv
import pickle

import accelerate
from huggingface_hub import hf_hub_download


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(0)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# %% [markdown]
# # Model

# %%
# from transformers import AutoConfig, LlamaForCausalLM, AutoTokenizer

# model_id = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
# model_name = "tinyllama"

# tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=MODEL_DIR)
# hf_model = LlamaForCausalLM.from_pretrained(
#     model_id, low_cpu_mem_usage=True, device_map='auto', cache_dir=MODEL_DIR,
#     torch_dtype=torch.bfloat16)
# hf_model = hf_model.eval()
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = 'left'

# VOCAB = sorted(tokenizer.vocab, key=tokenizer.vocab.get)

# layer_idx = 14

# %%
# Load model
from transformers import AutoModelForCausalLM, AutoTokenizer

with open("hf_token.txt", "r") as f:
    hf_token = f.read().strip()

model_id = "google/gemma-2-2b"
model_name = "gemma-2-2b"

torch.set_grad_enabled(False)  # avoid blowing up mem
hf_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    cache_dir=MODEL_DIR,
    token=hf_token,
    device_map=device,
    low_cpu_mem_usage=True,
    attn_implementation="eager",
)

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    cache_dir=MODEL_DIR,
    token=hf_token,
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
VOCAB = sorted(tokenizer.vocab, key=tokenizer.vocab.get)

layer_idx = 10

# %%
from nnsight import NNsight

nnsight_model = NNsight(hf_model)
nnsight_tracer_kwargs = {
    "scan": True,
    "validate": False,
    "use_cache": False,
    "output_attentions": False,
}

# %% [markdown]
# # Dataset

# %%
entity_type = "city"
INPUT_MAX_LEN = 48

# %%
import json
import os
import random

import datasets
from datasets import Dataset


FEATURE_TYPES = datasets.Features(
    {
        "input": datasets.Value("string"),
        "label": datasets.Value("string"),
        "source_input": datasets.Value("string"),
        "source_label": datasets.Value("string"),
        "inv_label": datasets.Value("string"),
        "split": datasets.Value("string"),
        "source_split": datasets.Value("string"),
        "entity": datasets.Value("string"),
        "source_entity": datasets.Value("string"),
    }
)


# Load training dataset.
split_to_raw_example = json.load(
    open(os.path.join(DATA_DIR, f"{model_name}/{model_name}_{entity_type}_train.json"), "r")
)
# Load validation + test dataset.
split_to_raw_example.update(
    json.load(
        open(
            os.path.join(DATA_DIR, f"{model_name}/{model_name}_{entity_type}_context_test.json"),
            "r",
        )
    )
)
split_to_raw_example.update(
    json.load(
        open(
            os.path.join(DATA_DIR, f"{model_name}/{model_name}_{entity_type}_entity_test.json"), "r"
        )
    )
)
# Prepend an extra token to avoid tokenization changes for Llama tokenizer.
# Each sequence will start with <s> _ 0
SOS_PAD = "0"
NUM_SOS_TOKENS = 3
for split in split_to_raw_example:
    for i in range(len(split_to_raw_example[split])):
        split_to_raw_example[split][i]["inv_label"] = (
            SOS_PAD + split_to_raw_example[split][i]["inv_label"]
        )
        split_to_raw_example[split][i]["label"] = SOS_PAD + split_to_raw_example[split][i]["label"]


# Load attributes (tasks) to prompt mapping.
ALL_ATTR_TO_PROMPTS = json.load(
    open(os.path.join(DATA_DIR, "base", f"ravel_{entity_type}_attribute_to_prompts.json"))
)

# Load prompt to intervention location mapping.
split_to_entity_pos = json.load(
    open(
        os.path.join(
            DATA_DIR, model_name, f"{model_name}_{entity_type}_prompt_to_entity_position.json"
        )
    )
)
SPLIT_TO_INV_LOCATIONS = {
    f"{task}{split}": {"max_input_length": INPUT_MAX_LEN, "inv_position": [INPUT_MAX_LEN + pos]}
    for task, pos in split_to_entity_pos.items()
    for split in ("-train", "-test", "-val", "")
}
assert min([min(v["inv_position"]) for v in SPLIT_TO_INV_LOCATIONS.values()]) > 0


# Preprocess the dataset.
def filter_inv_example(example):
    return (
        example["label"] != example["inv_label"]
        and example["source_split"] in SPLIT_TO_INV_LOCATIONS
        and example["split"] in SPLIT_TO_INV_LOCATIONS
    )


for split in split_to_raw_example:
    random.shuffle(split_to_raw_example[split])
    split_to_raw_example[split] = list(filter(filter_inv_example, split_to_raw_example[split]))
    if len(split_to_raw_example[split]) == 0:
        print('Empty split: "%s"' % split)
# Remove empty splits.
split_to_raw_example = {k: v for k, v in split_to_raw_example.items() if len(v) > 0}
print(
    f"#Training examples={sum(map(len, [v for k, v in split_to_raw_example.items() if k.endswith('-train')]))}, "
    f"#Validation examples={sum(map(len, [v for k, v in split_to_raw_example.items() if k.endswith('-val')]))}, "
    f"#Test examples={sum(map(len, [v for k, v in split_to_raw_example.items() if k.endswith('-test')]))}"
)
split_to_dataset = {
    split: Dataset.from_list(split_to_raw_example[split], features=FEATURE_TYPES)
    for split in split_to_raw_example
}

# #Training examples=116728, #Validation examples=20516, #Test examples=22497

# %% [markdown]
# # Sparse Autoencoder (SAE)

# %% [markdown]
# ## Tinyllama SAE Training
#
# We will train a sparse autoencoder on entity representations extracted offline.
#
# * Download entity representations extracted from the Wikipedia dataset [here](https://drive.google.com/file/d/1hZ-Nv3ehf0Ok4ic3ybe-DATEh-HRjYkt/view?usp=drive_link)

# %%
# from scripts.train_sae import train_sae
# import re

# config = {
#     'task_name': task_name,
#     'reg_coeff': float(re.search('reg([\d.]+)', task_name).group(1)),
#     'input_dim': model.config.hidden_size,
#     'latent_dim': int(re.search('dim(\d+)', task_name).group(1)),
#     'learning_rate': 1e-4,
#     'weight_decay': 1e-4,
#     'end_learning_rate_ratio': 0.5,
#     'num_epochs': int(re.search('ep(\d+)', task_name).group(1)),
#     'model_dir': MODEL_DIR,
#     'log_dir': os.path.join(MODEL_DIR, 'logs', task_name),
# }

# # Training metrics are logged to the Tensorboard at http://localhost:6006/.
# # autoencoder = train_sae(config, wiki_train_dataloader, wiki_val_dataloader)

# %%
# autoencoder_run_name = 'tinyllama-layer14-dim8192-reg0.5-ep5-sae-city_wikipedia_200k.pt'
# autoencoder = torch.load(os.path.join(MODEL_DIR, autoencoder_run_name)).to(device)

# %% [markdown]
# ## Gemma2 JumpRelu SAE

# %%


class JumpReluAutoEncoder(torch.nn.Module):
    """Sparse Autoencoder with a two-layer encoder and a two-layer decoder."""

    def __init__(self, embed_dim, latent_dim, device):
        super().__init__()
        self.dtype = torch.float32
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.W_enc = nn.Parameter(torch.empty(embed_dim, latent_dim))
        self.b_enc = nn.Parameter(torch.zeros(latent_dim))
        self.W_dec = nn.Parameter(torch.empty(latent_dim, embed_dim))
        self.b_dec = nn.Parameter(torch.zeros(embed_dim))
        self.threshold = nn.Parameter(torch.zeros(latent_dim))
        self.autoencoder_losses = {}

    def encode(self, x, normalize_input=False):
        if normalize_input:
            raise ValueError("Not supported")
            x = x - self.decoder[0].bias
        pre_jump = x @ self.W_enc + self.b_enc

        f = nn.ReLU()(pre_jump * (pre_jump > self.threshold))
        # Decoder weights are not normalized. Thus we have to compensate here to get comparabe feature activations.
        f = f * self.W_dec.norm(dim=1)
        return f

    def decode(self, z):
        # Decoder weights are not normalized. Thus we have to compensate here to get comparabe feature activations.
        z = z / self.W_dec.norm(dim=1)
        return z @ self.W_dec + self.b_dec

    def forward(self, base):
        base_type = base.dtype
        base = base.to(self.dtype)
        self.autoencoder_losses.clear()
        z = self.encode(base)
        base_reconstruct = self.decode(z)
        # The sparsity objective.
        l1_loss = torch.nn.functional.l1_loss(z, torch.zeros_like(z))
        # The reconstruction objective.
        l2_loss = torch.mean((base_reconstruct - base) ** 2)
        self.autoencoder_losses["l1_loss"] = l1_loss
        self.autoencoder_losses["l2_loss"] = l2_loss
        return {"latent": z, "output": base_reconstruct.to(base_type)}

    def get_autoencoder_losses(self):
        return self.autoencoder_losses

    def from_pretrained(
        path: str | None = None,
        load_from_sae_lens: bool = False,
        device: torch.device | None = None,
        **kwargs,
    ):
        """
        Load a pretrained autoencoder from a file.
        If sae_lens=True, then pass **kwargs to sae_lens's
        loading function.
        """
        state_dict = torch.load(path)
        latent_dim, embed_dim = state_dict["W_enc"].shape
        autoencoder = JumpReluAutoEncoder(embed_dim, latent_dim)
        autoencoder.load_state_dict(state_dict)


# %% [markdown]
# ## Feature Selection

# %%
# !cp data/tinyllama/ravel_city_tinyllama_layer14_representation.hdf5 data/ravel_city_tinyllama_layer14_representation.hdf5

# %%
# Load the RAVEL dataset.
import json

from src.utils.dataset_utils import load_entity_representation_with_label

splits = ["train", "val_entity", "val_context"]
feature_hdf5_path = os.path.join(
    DATA_DIR, model_name, f"ravel_{entity_type}_{model_name}_layer{layer_idx}_representation.hdf5"
)
entity_attr_to_label = json.load(
    open(os.path.join(DATA_DIR, "base", f"ravel_{entity_type}_entity_attributes.json"))
)
X, Y, sorted_unique_label = load_entity_representation_with_label(
    feature_hdf5_path, entity_attr_to_label, splits
)

# %%
# Run feature selection.
import numpy as np

from src.methods.select_features import select_features_with_classifier


# %%
# !wget -O tinyllama.tgz "https://huggingface.co/datasets/adamkarvonen/ravel/resolve/main/tinyllama.tgz?download=true"
# !tar -xzf tinyllama.tgz -C data/
# !mkdir data/base
# !tar -xvf data.tgz -C data/base --strip-components=1

# %% [markdown]
# ## Evaluation

# %% [markdown]
# ## VERY IMPORTANT NOTE
# In the below cell, we only use the first 10 elements of the dataset to speed up iteration. This should get increased once we have everything working.

# %%

# %% [markdown]
# ### Default pyvene implementation


# %%
def pyvene_intervention(
    intervenable,
    split_to_inv_locations,
    inputs,
    b_s,
    num_inv,
    max_new_tokens,
    intervention_locations,
    forward_only=False,
):
    if not forward_only:
        base_outputs, counterfactual_out_tokens = intervenable.generate(
            {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
            },
            [
                {
                    "input_ids": inputs["source_input_ids"],
                    "attention_mask": inputs["source_attention_mask"],
                    "position_ids": inputs["source_position_ids"],
                }
            ],
            intervention_locations,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            intervene_on_prompt=True,
            pad_token_id=tokenizer.pad_token_id,
            output_original_output=True,
        )
    # else: # This seems deprecated in the demo notebook
    #     base_outputs, counterfactual_outputs = intervenable(
    #         {
    #             "input_ids": inputs["input_ids"],
    #             "attention_mask": inputs["attention_mask"],
    #             "position_ids": inputs["position_ids"],
    #         },
    #         [
    #             {
    #                 "input_ids": inputs["source_input_ids"],
    #                 "attention_mask": inputs["source_attention_mask"],
    #                 "position_ids": inputs["source_position_ids"],
    #             }
    #         ],
    #         intervention_locations,
    #         output_original_output=True,
    #     )
    #     counterfactual_logits = counterfactual_outputs.logits
    #     counterfactual_out_tokens = torch.argmax(counterfactual_outputs.logits, dim=-1)
    #     base_outputs = torch.argmax(base_outputs.logits, dim=-1)

    return base_outputs, counterfactual_out_tokens


# %% [markdown]
# ### nnsight replication

# %%
import einops
import torch


def nnsight_intervention(
    nnsight_model,
    layer,
    autoencoder,
    inv_dims,
    inputs,
    split_to_inv_locations,
    n_generated_tokens,
    device,
    add_reconstruction_error=True,
    inv_positions=None,
    verbose=False,
):
    batch_size = inputs["input_ids"].shape[0]
    submodule = nnsight_model.model.layers[layer]

    # Organize intervention positions
    if inv_positions is None:  # Default, custom intervention positions only for testing
        base_inv_positions = torch.tensor(
            [split_to_inv_locations[inputs["split"][i]]["inv_position"] for i in range(batch_size)],
            device=device,
        )
        source_inv_positions = torch.tensor(
            [
                split_to_inv_locations[inputs["source_split"][i]]["inv_position"]
                for i in range(batch_size)
            ],
            device=device,
        )
    else:
        base_inv_positions, source_inv_positions = inv_positions

    # Indexing preparation
    if isinstance(inv_dims, range):
        inv_dims = torch.tensor(list(inv_dims), device=device, dtype=torch.int)
    if len(base_inv_positions.shape) > 1:
        base_inv_positions = base_inv_positions.squeeze(dim=-1)
    if len(source_inv_positions.shape) > 1:
        source_inv_positions = source_inv_positions.squeeze(dim=-1)

    batch_arange = einops.repeat(
        torch.arange(batch_size, device=device, dtype=torch.int), "b -> b d", d=inv_dims.shape[0]
    )
    base_inv_positions = einops.repeat(base_inv_positions, "b -> b d", d=inv_dims.shape[0])
    source_inv_positions = einops.repeat(source_inv_positions, "b -> b d", d=inv_dims.shape[0])
    inv_dims = einops.repeat(inv_dims, "d -> b d", b=batch_size)

    # Forward pass on source input
    with torch.no_grad(), nnsight_model.trace(
        inputs["source_input_ids"],
        attention_mask=inputs["source_attention_mask"],
        **nnsight_tracer_kwargs,
    ):
        source_sae_acts = autoencoder.encode(submodule.output[0])
        source_sae_acts = source_sae_acts[batch_arange, source_inv_positions, inv_dims].save()

    # Forward pass on base input with intervention
    generated_inputs_shape = inputs["input_ids"].shape[:-1] + (
        inputs["input_ids"].shape[-1] + n_generated_tokens,
    )
    generated_inputs = torch.zeros(
        generated_inputs_shape, device=device, dtype=inputs["input_ids"].dtype
    )
    generated_inputs[:, : inputs["input_ids"].shape[-1]] = inputs["input_ids"]
    generated_attn_mask = torch.cat(
        [inputs["attention_mask"], torch.ones(batch_size, n_generated_tokens, device=device)],
        dim=-1,
    )
    n_tokens = generated_attn_mask.sum(dim=-1).to(torch.int)
    generated_pos_ids = torch.zeros_like(
        generated_inputs, device=device, dtype=inputs["input_ids"].dtype
    )
    for batch_idx in range(batch_size):
        generated_pos_ids[batch_idx, -n_tokens[batch_idx] :] = torch.arange(
            n_tokens[batch_idx], device=device, dtype=inputs["input_ids"].dtype
        )

    for i in range(n_generated_tokens):
        with torch.no_grad(), nnsight_model.trace(
            generated_inputs,
            attention_mask=generated_attn_mask,
            position_ids=generated_pos_ids,
            **nnsight_tracer_kwargs,
        ):
            llm_acts = submodule.output[0]
            base_sae_acts = autoencoder.encode(llm_acts)
            llm_acts_reconstructed = autoencoder.decode(base_sae_acts)

            base_sae_acts[batch_arange, base_inv_positions, inv_dims] = source_sae_acts
            llm_acts_intervened = autoencoder.decode(base_sae_acts)

            if not add_reconstruction_error:
                submodule.output = (llm_acts_intervened.to(llm_acts.dtype),)
            else:
                reconstruction_error = llm_acts - llm_acts_reconstructed
                corrected_acts = llm_acts_intervened + reconstruction_error
                submodule.output = (corrected_acts.to(llm_acts.dtype),)

            counterfactual_logits = nnsight_model.lm_head.output.save()

        # Append generation
        final_token_pos = i - n_generated_tokens
        next_token = torch.argmax(counterfactual_logits[:, final_token_pos - 1, :], dim=-1)
        generated_inputs[:, final_token_pos] = next_token

        if verbose:
            print(
                f"counterfactual_out_tokens decoded: {tokenizer.batch_decode(generated_inputs, skip_special_tokens=True)}"
            )

    return generated_inputs


# Test the intervention

# batch_size = 2
# intervention_dims = range(autoencoder.latent_dim)
# base_promts = ["Paris is in the country of", "The main language spoken in the city of London is"]
# source_prompts = ["Tokyo is big.", "Berlin is exciting."]
# n_generated_tokens = 3
# add_reconstruction_error = True

# base_tok = tokenizer(
#     base_promts,
#     return_tensors="pt",
#     padding="max_length",
#     truncation=True,
#     max_length=INPUT_MAX_LEN,
# )
# source_tok = tokenizer(
#     source_prompts,
#     return_tensors="pt",
#     padding="max_length",
#     truncation=True,
#     max_length=INPUT_MAX_LEN,
# )
# base_tok = hf_model.prepare_inputs_for_generation(**base_tok)
# source_tok = hf_model.prepare_inputs_for_generation(**source_tok)
# base_inv_pos = torch.ones(batch_size, device=device, dtype=torch.int) * -1
# source_inv_pos = torch.ones(batch_size, device=device, dtype=torch.int) * -1
# base_inv_pos = torch.tensor([[42], [46]], dtype=torch.int)

# # # Tinyllama
# # source_inv_pos = torch.tensor([[44], [43]], dtype=torch.int)

# # Gemma2-2b
# source_inv_pos = torch.tensor([[44], [44]], dtype=torch.int)

# # print(f'base_inv_pos: {base_inv_pos}')
# # print(f'source_inv_pos: {source_inv_pos}')

# # for tok_ids in source_tok['input_ids']:
# #     for j, tok in enumerate(tok_ids):
# #         print(f'{j}, tok: {tokenizer.decode(tok)}')


# # for i in torch.arange(1, len(base_tok['input_ids'][0]), device=device, dtype=torch.int) * -1:
# # if i == 10:
# #         break
# #     print(f'i: {i}')
# #     base_inv_pos = torch.tensor([[i]])
# print(
#     f'prompt1 base_token decoded: {tokenizer.decode(base_tok["input_ids"][0][base_inv_pos[0][0]])}'
# )
# print(
#     f'prompt1 source_token decoded: {tokenizer.decode(source_tok["input_ids"][0][source_inv_pos[0][0]])}'
# )
# print(
#     f'prompt2 base_token decoded: {tokenizer.decode(base_tok["input_ids"][1][base_inv_pos[1][0]])}'
# )
# print(
#     f'prompt2 source_token decoded: {tokenizer.decode(source_tok["input_ids"][1][source_inv_pos[1][0]])}'
# )

# inputs = {
#     "input_ids": base_tok["input_ids"].to(device),
#     "attention_mask": base_tok["attention_mask"].to(device),
#     "position_ids": base_tok["position_ids"].to(device),
#     "source_input_ids": source_tok["input_ids"].to(device),
#     "source_attention_mask": source_tok["attention_mask"].to(device),
#     "source_position_ids": source_tok["position_ids"].to(device),
# }


# counterfactual_out_tokens = nnsight_intervention(
#     nnsight_model,
#     layer_idx,
#     autoencoder,
#     intervention_dims,
#     inputs,
#     None,  # split_to_inv_locations is not used in the test
#     n_generated_tokens,
#     device,
#     add_reconstruction_error,
#     inv_positions=(base_inv_pos, source_inv_pos),
#     verbose=True,
# )


# %%
from src.utils.intervention_utils import (
    is_llama_tokenizer,
    get_dataloader,
    remove_invalid_token_id,
    load_intervenable_with_autoencoder,
    remove_all_forward_hooks,
)
import collections
from tqdm import tqdm


def eval_with_interventions(
    hf_model,  # Native Hugging Face model
    nnsight_model,  # NNsight model wrapper
    split_to_dataset,
    split_to_inv_locations,
    tokenizer: AutoTokenizer,
    inv_dims,
    compute_metrics_fn,
    max_new_tokens=1,
    eval_batch_size=16,
    debug_print=False,
    forward_only=False,
    use_nnsight_replication=False,
    device="cuda",
):
    if not use_nnsight_replication:
        intervenable = load_intervenable_with_autoencoder(
            hf_model, autoencoder, inv_dims, layer_idx
        )
        intervenable.set_device("cuda")
        intervenable.disable_model_gradients()
        num_inv = len(intervenable.interventions)
    else:
        num_inv = 1

    split_to_eval_metrics = {}
    padding_offset = 3 if is_llama_tokenizer(tokenizer) else 0
    for split in tqdm(split_to_dataset, desc="Splits"):
        print(f"Split: {split}")
        # Asssume all inputs have the same max length.
        prompt_max_length = split_to_inv_locations[split_to_dataset[split][0]["split"]][
            "max_input_length"
        ]
        eval_dataloader = get_dataloader(
            split_to_dataset[split],
            tokenizer=tokenizer,
            batch_size=eval_batch_size,
            prompt_max_length=prompt_max_length,
            output_max_length=padding_offset + max_new_tokens,
            first_n=max_new_tokens,
        )
        eval_labels = collections.defaultdict(list)
        eval_preds = []
        with torch.no_grad():
            if debug_print:
                epoch_iterator = tqdm(eval_dataloader, desc=f"Test")
            else:
                epoch_iterator = eval_dataloader
            for step, inputs in enumerate(epoch_iterator):
                b_s = inputs["input_ids"].shape[0]
                position_ids = {
                    f"{prefix}position_ids": hf_model.prepare_inputs_for_generation(
                        input_ids=inputs[f"{prefix}input_ids"],
                        attention_mask=inputs[f"{prefix}attention_mask"],
                    )["position_ids"]
                    for prefix in ("", "source_")
                }
                inputs.update(position_ids)
                for key in inputs:
                    if key in (
                        "input_ids",
                        "source_input_ids",
                        "attention_mask",
                        "source_attention_mask",
                        "position_ids",
                        "source_position_ids",
                        "labels",
                        "base_labels",
                    ):
                        inputs[key] = inputs[key].to(hf_model.device)

                intervention_locations = {
                    "sources->base": (
                        [
                            [
                                split_to_inv_locations[inputs["source_split"][i]]["inv_position"]
                                for i in range(b_s)
                            ]
                        ]
                        * num_inv,
                        [
                            [
                                split_to_inv_locations[inputs["split"][i]]["inv_position"]
                                for i in range(b_s)
                            ]
                        ]
                        * num_inv,
                    )
                }

                if not use_nnsight_replication:
                    base_outputs, counterfactual_out_tokens = pyvene_intervention(
                        intervenable,
                        split_to_inv_locations,
                        inputs,
                        b_s,
                        num_inv,
                        max_new_tokens,
                        forward_only=forward_only,
                    )
                    eval_preds.append(counterfactual_out_tokens)
                else:
                    base_outputs = hf_model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_length=inputs["input_ids"].shape[1] + max_new_tokens,
                        pad_token_id=tokenizer.pad_token_id,
                        num_beams=1,
                        do_sample=False,
                        output_scores=True,
                    )
                    counterfactual_out_tokens = nnsight_intervention(
                        nnsight_model,
                        layer_idx,
                        autoencoder,
                        inv_dims,
                        inputs,
                        split_to_inv_locations,
                        n_generated_tokens=max_new_tokens,
                        device=device,
                        add_reconstruction_error=True,
                    )
                    eval_preds.append(counterfactual_out_tokens)

                for label_type in ["base_labels", "labels"]:
                    eval_labels[label_type].append(inputs[label_type])
                eval_labels["base_outputs"].append(base_outputs[:, -max_new_tokens:])

                if debug_print and step < 3:
                    print("\nInputs:")
                    print("Base:", inputs["input"][:3])
                    print("Source:", inputs["source_input"][:3])
                    print("Tokens to intervene:")
                    print(
                        "Base:",
                        tokenizer.batch_decode(
                            [
                                inputs["input_ids"][i][
                                    intervention_locations["sources->base"][1][0][i]
                                ]
                                for i in range(len(inputs["split"]))
                            ]
                        ),
                    )
                    print(
                        "Source:",
                        tokenizer.batch_decode(
                            [
                                inputs["source_input_ids"][i][
                                    intervention_locations["sources->base"][0][0][i]
                                ]
                                for i in range(len(inputs["split"]))
                            ]
                        ),
                    )
                    base_output_text = tokenizer.batch_decode(
                        base_outputs[:, -max_new_tokens:], skip_special_tokens=True
                    )
                    print("Base Output:", base_output_text)
                    print(
                        "Output:    ",
                        tokenizer.batch_decode(counterfactual_out_tokens[:, -max_new_tokens:]),
                    )
                    print(
                        "Inv Label: ",
                        tokenizer.batch_decode(
                            remove_invalid_token_id(
                                inputs["labels"][:, :max_new_tokens], tokenizer.pad_token_id
                            )
                        ),
                    )
                    base_label_text = tokenizer.batch_decode(
                        remove_invalid_token_id(
                            inputs["base_labels"][:, :max_new_tokens], tokenizer.pad_token_id
                        ),
                        skip_special_tokens=True,
                    )
                    print("Base Label:", base_label_text)
                    if base_label_text != base_output_text:
                        print("WARNING: Base outputs does not match base labels!")

        eval_metrics = {
            label_type: compute_metrics_fn(
                tokenizer,
                eval_preds,
                eval_labels[label_type],
                last_n_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                extra_labels=eval_labels,
                eval_label_type=label_type,
            )
            for label_type in eval_labels
            if label_type.endswith("labels")
        }
        print("\n", repr(split) + ":", eval_metrics)
        split_to_eval_metrics[split] = {
            "metrics": eval_metrics,
            "inv_outputs": tokenizer.batch_decode(counterfactual_out_tokens[:, -max_new_tokens:]),
            "inv_labels": tokenizer.batch_decode(
                remove_invalid_token_id(
                    inputs["labels"][:, :max_new_tokens], tokenizer.pad_token_id
                )
            ),
            "base_labels": tokenizer.batch_decode(
                remove_invalid_token_id(
                    inputs["base_labels"][:, :max_new_tokens], tokenizer.pad_token_id
                )
            ),
        }

    if not use_nnsight_replication:
        remove_all_forward_hooks(intervenable)
        del intervenable
    return split_to_eval_metrics


# %%
def compute_metrics(tokenizer, eval_preds, eval_labels, pad_token_id, last_n_tokens=1, **kwargs):
    """Computes squence-level and token-level accuracy."""
    total_count, total_token_count = 0, 0
    correct_count, correct_token_count = 0, 0
    for eval_pred, eval_label in zip(eval_preds, eval_labels):
        actual_test_labels = eval_label[:, -last_n_tokens:]
        if len(eval_pred.shape) == 3:
            # eval_preds is in the form of logits.
            pred_test_labels = torch.argmax(eval_pred[:, -last_n_tokens:], dim=-1)
        else:
            # eval_preds is in the form of token ids.
            pred_test_labels = eval_pred[:, -last_n_tokens:]
        padding_tokens = torch.logical_or(
            actual_test_labels == pad_token_id, actual_test_labels < 0
        )
        match_tokens = actual_test_labels == pred_test_labels
        correct_labels = torch.logical_or(match_tokens, padding_tokens)
        total_count += len(correct_labels)
        correct_count += torch.all(correct_labels, axis=-1).float().sum().tolist()
        total_token_count += (~padding_tokens).float().sum().tolist()
        correct_token_count += (~padding_tokens & match_tokens).float().sum().tolist()
    accuracy = round(correct_count / total_count, 2)
    token_accuracy = round(correct_token_count / total_token_count, 2)
    return {"accuracy": accuracy, "token_accuracy": token_accuracy}


def compute_metrics_string_matching(tokenizer, eval_preds, eval_labels, last_n_tokens, **kwargs):
    """Computes squence-level and string-level accuracy."""
    total_count = 0
    correct_count = 0

    eval_preds = torch.cat(eval_preds, dim=0)
    eval_labels = torch.cat(eval_labels, dim=0)[:, -last_n_tokens:]
    eval_preds_str = tokenizer.batch_decode(eval_preds, skip_special_tokens=True)
    eval_labels_str = tokenizer.batch_decode(eval_labels, skip_special_tokens=True)
    eval_labels_str = [l[1:].strip() for l in eval_labels_str]

    for p, l in zip(eval_preds_str, eval_labels_str):
        # print(f'p: {p}, l: {l}')
        total_count += 1
        if l.lower() in p.lower():
            correct_count += 1

    accuracy = round(correct_count / total_count, 2)
    return {"accuracy": accuracy}


def save_eval_metrics(split_to_eval_metrics: dict, logging_filename: str):
    try:
        # Check if the file already exists
        if os.path.exists(logging_filename):
            # If it exists, load the existing data
            with open(logging_filename, "r") as f:
                existing_data = json.load(f)

            # Update the existing data with new data
            existing_data.update(split_to_eval_metrics)
        else:
            # If it doesn't exist, use the new data as is
            existing_data = split_to_eval_metrics

        # Write the updated or new data to the file
        with open(logging_filename, "w") as f:
            json.dump(existing_data, f, indent=4)

        print(f"Successfully updated/saved JSON file: {logging_filename}")

    except (TypeError, ValueError, OSError) as e:
        print(f"JSON dump failed: {e}. Attempting fallback options.")
        backup_filename = logging_filename.replace(".json", "_backup.json")
        print(f"Attempting to save to backup file: {backup_filename}")
        with open(backup_filename, "w") as f:
            json.dump(split_to_eval_metrics, f, indent=4)


def convert_inv_dims_to_list(inv_dims):
    if isinstance(inv_dims, range):
        inv_dims = list(inv_dims)
    if isinstance(inv_dims, torch.Tensor):
        inv_dims = inv_dims.tolist()
    if isinstance(inv_dims, np.ndarray):
        inv_dims = inv_dims.tolist()
    return inv_dims


def verify_inv_dim_saving(inv_dims):
    temp_dict = {}
    for inv_name, inv_dims in intervention_dim_to_eval:
        print("inv_name:", inv_name)
        temp_dict[inv_name] = convert_inv_dims_to_list(inv_dims)
    with open("intervention_dim_to_eval.json", "w") as f:
        json.dump(temp_dict, f)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def load_autoencoder(sae_repo_id: str, sae_filename: str, model_name: str, device: str):
    path_to_params = hf_hub_download(
        repo_id=sae_repo_id,
        filename=sae_filename,
        force_download=False,
        cache_dir=os.path.join(MODEL_DIR, model_name),
    )
    params = np.load(path_to_params)
    pt_params = {k: torch.from_numpy(v).cuda() for k, v in params.items()}

    embed_dim = params["W_enc"].shape[0]
    latent_dim = params["W_enc"].shape[1]

    sae = JumpReluAutoEncoder(
        embed_dim=embed_dim,
        latent_dim=latent_dim,
        device=device,
    )
    sae.load_state_dict(pt_params)
    sae.to(device)

    return sae


# %%
torch.cuda.empty_cache()

# %%
use_nnsight_replication = True

# Run eval
import re
import gc

sae_repo_id = "google/gemma-scope-2b-pt-res"
# sae_filename = "layer_20/width_16k/average_l0_71/params.npz"
sae_filename = "layer_14/width_16k/average_l0_43/params.npz"
sae_filename = "layer_10/width_16k/average_l0_77/params.npz"


l0s = [21, 39, 77, 166, 395]

feature_selection_coefficients = [0.1, 10, 100, 1000]
feature_selection_coefficients = [0.005, 0.01, 0.05]

layer_idx = 10
max_dataset_size = 10

# verify that l0s exist
for l0 in l0s:
    sae_filename = f"layer_{layer_idx}/width_16k/average_l0_{l0}/params.npz"
    autoencoder = load_autoencoder(sae_repo_id, sae_filename, model_name, device)
    del autoencoder

for l0 in l0s:
    gc.collect()
    torch.cuda.empty_cache()
    sae_filename = f"layer_{layer_idx}/width_16k/average_l0_{l0}/params.npz"

    autoencoder_run_name = (sae_repo_id + "-" + sae_filename.replace("/", "-")).replace(".npz", "")

    autoencoder = load_autoencoder(sae_repo_id, sae_filename, model_name, device)

    intervention_dim_to_eval = [
        ("reconstruction", None),
        ("dim%d" % autoencoder.latent_dim, range(autoencoder.latent_dim)),
    ]

    intervention_dim_to_eval = []

    attr = "Country"
    coeff_to_kept_dims = select_features_with_classifier(
        autoencoder.encode,
        torch.from_numpy(X[attr]["train"]).to(device),
        Y[attr]["train"],
        coeff=feature_selection_coefficients,
    )
    for kept_dim in coeff_to_kept_dims.values():
        intervention_dim_to_eval.append(("dim%d" % len(kept_dim), kept_dim))
    # Random baselines.
    # for i in [64, 512]:
    #     kept_dim = np.random.permutation(autoencoder.latent_dim)[:i]
    #     intervention_dim_to_eval.append(("random_dim%d" % len(kept_dim), kept_dim))

    verify_inv_dim_saving(intervention_dim_to_eval)

    eval_split_to_dataset = {
        k: v for k, v in split_to_dataset.items() if k.endswith("-test") or k.endswith("-val")
    }
    print(len(eval_split_to_dataset))

    # # Keep only the first 10 items
    eval_split_to_dataset = dict(list(eval_split_to_dataset.items())[:max_dataset_size])
    print(f"New length: {len(eval_split_to_dataset)}")

    print(len(eval_split_to_dataset))

    target_task = "Country"
    max_new_tokens = 3
    print(f"Layer={layer_idx}")

    for inv_name, inv_dims in tqdm(intervention_dim_to_eval, desc="Intervention dim"):
        print(f"Inv name={inv_name}, l0={l0}")
        if inv_name == "reconstruction":
            continue
        print(f"Intervention_dims={inv_dims}")
        split_to_eval_metrics = eval_with_interventions(
            hf_model=hf_model,
            nnsight_model=nnsight_model,
            split_to_dataset=eval_split_to_dataset,
            split_to_inv_locations=SPLIT_TO_INV_LOCATIONS,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            inv_dims=inv_dims,
            compute_metrics_fn=compute_metrics_string_matching,
            eval_batch_size=32,  # batchsize=32 yields 21GB RAM usage with RTX A6000
            debug_print=False,
            use_nnsight_replication=use_nnsight_replication,
            device=device,
        )
        split_to_eval_metrics["hyperparameters"] = {}
        split_to_eval_metrics["hyperparameters"]["l0"] = l0
        split_to_eval_metrics["hyperparameters"]["layer_idx"] = layer_idx
        split_to_eval_metrics["hyperparameters"]["inv_name"] = inv_name
        split_to_eval_metrics["hyperparameters"]["inv_dims"] = convert_inv_dims_to_list(inv_dims)

        logging_filename = f'{autoencoder_run_name.split(".pt")[0]}_{inv_name}_{max_new_tokens}tok_{target_task}_dataset_size_{max_dataset_size}.json'
        if "/" in logging_filename:
            logging_filename = logging_filename.replace("/", "-")

        ae_folder = f"layer{layer_idx}_l0{l0}"
        full_path = os.path.join(MODEL_DIR, ae_folder)
        if not os.path.exists(full_path):
            os.makedirs(full_path)

        logging_filename = os.path.join(full_path, logging_filename)

        save_eval_metrics(split_to_eval_metrics, logging_filename)


raise ValueError("Stop here")

# %% [markdown]
# # Distributed Alignment Search (DAS/MDAS)

# %% [markdown]
# ## Training

# %%
import collections
import numpy as np
import re

from datasets import concatenate_datasets
from src.methods.distributed_alignment_search import LowRankRotatedSpaceIntervention
from src.methods.differential_binary_masking import DifferentialBinaryMasking
import pyvene as pv
from tqdm import tqdm, trange
from scripts.train_intervention import train_intervention
from transformers import get_linear_schedule_with_warmup
from src.utils.dataset_utils import get_multitask_dataloader
from src.utils.intervention_utils import (
    train_intervention_step,
    eval_with_interventions,
    get_intervention_config,
    remove_all_forward_hooks,
    remove_invalid_token_id,
)
from src.utils.metric_utils import compute_metrics, compute_cross_entropy_loss


def get_short_model_name(model):
    name_match = re.search("(llama-2-\d+b|tinyllama|pythia-[\d.]+b)", model.name_or_path.lower())
    if name_match:
        return name_match.group(1)
    else:
        return model.name_or_path.lower().split("-")[0]


def run_exp(config):
    inv_tasks = "+".join(
        [
            "".join(re.findall(r"[A-Za-z]+", t))
            for t, l in config["training_tasks"].items()
            if "match_source" in l
        ]
    )
    control_tasks = "+".join(
        [
            "".join(re.findall(r"[A-Za-z]+", t))
            for t, l in config["training_tasks"].items()
            if "match_base" in l
        ]
    )
    task_compressed = (inv_tasks + "_ex_" + control_tasks) if control_tasks else inv_tasks
    method_name = "multitask_method" if len(config["training_tasks"]) > 1 else "baseline_method"
    if (
        config["intervenable_config"]["intervenable_interventions_type"]
        == LowRankRotatedSpaceIntervention
    ):
        method_name = method_name.replace("method", "daslora")
    elif (
        config["intervenable_config"]["intervenable_interventions_type"]
        == DifferentialBinaryMasking
    ):
        if config["regularization_coefficient"] > 1e-6:
            method_name = method_name.replace("method", "mask_l1")
        else:
            method_name = method_name.replace("method", "mask")
    split_to_inv_locations = config["split_to_inv_locations"]
    input_len = list(split_to_inv_locations.values())[0]["max_input_length"]
    inv_pos = min([x["inv_position"][0] for x in split_to_inv_locations.values()])
    inv_loc_name = "len%d_pos%s" % (input_len, "e" if inv_pos != input_len - 1 else "f")
    training_data_percentage = int(config["max_train_percentage"] * 100)
    suffix = f"_cause{config['cause_task_sample_size']}"
    if any([v == "match_base" for t, v in config["training_tasks"].items()]):
        suffix += f'_iso{config["iso_task_sample_size"]}'
    layer = config["intervenable_config"]["intervenable_layer"]
    run_name = (
        f"{get_short_model_name(model)}-layer{layer}"
        f"-dim{config['intervention_dimension']}"
        f"-{method_name}_{config['max_output_tokens']}tok_"
        f"{task_compressed}_{inv_loc_name}_ep{config['training_epoch']}{suffix}"
    )
    config["run_name_prefix"] = run_name.rsplit("_ep", 1)[0]
    print(run_name)
    intervenable, intervenable_config = train_intervention(
        config, model, tokenizer, split_to_dataset
    )
    # Save model.
    torch.save(
        {k: v[0].rotate_layer.weight for k, v in intervenable.interventions.items()},
        os.path.join(MODEL_DIR, f"{run_name}.pt"),
    )
    print("Model saved to %s" % os.path.join(MODEL_DIR, f"{run_name}.pt"))
    # Eval.
    split_to_eval_metrics = eval_with_interventions(
        intervenable,
        eval_split_to_dataset,
        split_to_inv_locations,
        tokenizer,
        compute_metrics_fn=compute_metrics,
        max_new_tokens=config["max_output_tokens"],
        eval_batch_size=EVAL_BATCH_SIZE,
    )
    # Logging.
    json.dump(split_to_eval_metrics, open(os.path.join(MODEL_DIR, f"{run_name}_evalall.json"), "w"))
    print("Saved to %s" % os.path.join(MODEL_DIR, f"{run_name}_evalall.json"))
    remove_all_forward_hooks(intervenable)
    return intervenable


attrs = list(ALL_ATTR_TO_PROMPTS)
target_attr = "Country"

# Train on disentangling Country attribute only.
training_tasks_list = [{t: "match_source"} for t in attrs if t == target_attr] + [
    {t: "match_source" if t == target_t else "match_base" for t in attrs}
    for target_t in attrs
    if target_t == target_attr
]

# eval_split_to_dataset = {k: v for k, v in split_to_dataset.items()
#                          if k.endswith('-test') or k.endswith('-val')}
print(len(training_tasks_list))
print(training_tasks_list)

# %%

model = model.eval()

TRAINING_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 128

lr = 1e-4
for inv_layer in [14]:
    for inv_dim in [64]:
        for training_tasks in training_tasks_list:
            for cause_task_sample_size in [20000]:
                config = {
                    "regularization_coefficient": 0,
                    "intervention_dimension": inv_dim,
                    "max_output_tokens": 3,
                    "intervenable_config": {
                        "intervenable_layer": inv_layer,
                        "intervenable_representation_type": "block_output",
                        "intervenable_unit": "pos",
                        "max_number_of_units": 1,
                        "intervenable_interventions_type": LowRankRotatedSpaceIntervention,
                    },
                    "training_tasks": training_tasks,
                    "training_epoch": 3,
                    "split_to_inv_locations": SPLIT_TO_INV_LOCATIONS,
                    "max_train_percentage": 1.0 if len(training_tasks) <= 3 else 1.0,
                    "init_lr": lr,
                    "cause_task_sample_size": cause_task_sample_size,
                    "iso_task_sample_size": 4000,
                    "training_batch_size": TRAINING_BATCH_SIZE,
                    "task_to_prompts": ALL_ATTR_TO_PROMPTS,
                    "log_dir": os.path.join(MODEL_DIR, "logs"),
                }
                intervenable = run_exp(config)


# Training each method will take about 3.5 hrs on the hosted T4 runtime.

# %% [markdown]
# ## Evaluate

# %%
# # The training script above has already included the evaluation part.
# # Below is a standalone evaluation script in case you want to rerun evaluation.


# import re

# import pyvene as pv
# from src.utils.intervention_utils import load_intervenable, load_intervenable_with_pca, eval_with_interventions
# from src.utils.metric_utils import compute_metrics


# model_paths = [
#     'tinyllama-layer14-dim64-multitask_daslora_3tok_Country_ex_Continent+Latitude+Longitude+Language+Timezone_len48_pose_ep3_cause20000_iso4000.pt',
#     'tinyllama-layer14-dim64-baseline_daslora_3tok_Country_len48_pose_ep3_cause20000.pt',
#  ]

# eval_split_to_dataset = {k: v for k, v in split_to_dataset.items()
#                          if k.endswith('-test')
#                          }
# RUN_TO_EVAL_METRICS = {}
# for i, run_name in enumerate(model_paths):
#   print(run_name)
#   layer = int(re.search('layer(\d+)[_\-]', run_name).group(1))
#   run_name, ext = run_name.rsplit('.', 1)
#   if 'pca' in run_name:
#     intervenable = load_intervenable_with_pca(model, run_name + '.' + ext)
#   elif 'causal_abstraction' in run_name:
#     # NOTE: This is not available
#     intervenable = load_causal_abstraction_intervenable(model, run_name)
#   else:
#     intervenable = load_intervenable(model, os.path.join(MODEL_DIR, run_name + '.' + ext))
#   split_to_eval_metrics = eval_with_interventions(
#       intervenable, eval_split_to_dataset, SPLIT_TO_INV_LOCATIONS if layer < 24 else SPLIT_TO_INV_LOCATIONS_LAST_TOK,
#       tokenizer, compute_metrics_fn=compute_metrics, max_new_tokens=3, debug_print=False)
#   json.dump(split_to_eval_metrics, open(os.path.join(MODEL_DIR, f'{run_name}_evalall.json'), 'w'))

# %% [markdown]
# # Compare Methods with Disentangle Score

# %%
# Compute disentangle scores.

from src.utils.metric_utils import (
    compute_disentangle_score,
    compute_disentangle_scores_possible_empties,
)


tinyllama_dimension_to_log_path = {
    "SAE": {
        d: f"tinyllama-layer14-dim8192-reg0.5-ep5-sae-city_wikipedia_200k_dim{d}_3tok_Country.json"
        # Update the following dimensions to match your own results.
        # SAE might have different feature dimensions from run to run due to
        # randomness in the feature selection algorithm.
        for d in [71, 325, 399, 536, 8192]
    },
    # 'DAS': {d: f'tinyllama-layer14-dim{d}-baseline_daslora_3tok_Country_len48_pose_ep3_cause20000_evalall.json'
    #         for d in [16, 64]},
    # 'MDAS': {d: f'tinyllama-layer14-dim{d}-multitask_daslora_3tok_Country1_ex_Continent+Latitude+Longitude+Language+Timezone_len48_pose_ep3_cause20000_iso4000_evalall.json'
    #          for d in [16, 64]
    # },
}
# tinyllama_dimension_to_log_path['RandomSAE'] = {
#             64: f'tinyllama-layer14-dim8192-reg0.5-ep5-sae-city_wikipedia_200k_random_dim64_3tok_Country.json',
#             512: f'tinyllama-layer14-dim8192-reg0.5-ep5-sae-city_wikipedia_200k_random_dim512_3tok_Country.json'
#             }

entity_type = "city"
target_attribute = "Country"
split_type = "context"
split_suffix = "-test"
model_name = "tinyllama"


split_to_raw_example = json.load(
    open(os.path.join(DATA_DIR, model_name, f"{model_name}_{entity_type}_{split_type}_test.json"))
)
attribute_to_prompts = json.load(
    open(os.path.join(DATA_DIR, "base", f"ravel_{entity_type}_attribute_to_prompts.json"))
)


attribute_to_iso_tasks = {
    a: [p + split_suffix for p in ps if p + split_suffix in split_to_raw_example]
    for a, ps in attribute_to_prompts.items()
    if a != target_attribute
}
attribute_to_cause_tasks = {
    a: [p + split_suffix for p in ps if p + split_suffix in split_to_raw_example]
    for a, ps in attribute_to_prompts.items()
    if a == target_attribute
}

print(attribute_to_iso_tasks)

for key in attribute_to_iso_tasks:
    print(key)

for key in attribute_to_cause_tasks:
    print(key, "F")


method_to_data = collections.defaultdict(dict)
for method in tinyllama_dimension_to_log_path:
    for inv_dimension in tinyllama_dimension_to_log_path[method]:
        log_data = json.load(
            open(os.path.join(MODEL_DIR, tinyllama_dimension_to_log_path[method][inv_dimension]))
        )

        # print(log_data)

        # for key in log_data:
        #   print(key, log_data[key], "\n")

        method_to_data[method][inv_dimension] = compute_disentangle_scores_possible_empties(
            log_data, attribute_to_iso_tasks, attribute_to_cause_tasks
        )

# %%
print(method_to_data)

# %%
import matplotlib.pyplot as plt
import numpy as np

# Assuming method_to_data is defined elsewhere in your code
# method_to_data = defaultdict(<class 'dict'>, {'SAE': {71: {'disentangle': 0.41708333333333336, 'isolate': 0.49666666666666665, 'cause': 0.3375}, 325: {'disentangle': 0.4729166666666667, 'isolate': 0.06333333333333334, 'cause': 0.8825000000000001}, 399: {'disentangle': 0.4741666666666667, 'isolate': 0.05333333333333334, 'cause': 0.895}, 536: {'disentangle': 0.4779166666666667, 'isolate': 0.06333333333333334, 'cause': 0.8925000000000001}, 8192: {'disentangle': 0.47833333333333333, 'isolate': 0.06666666666666667, 'cause': 0.89}}})

# Extract data for SAE method
sae_data = method_to_data["SAE"]

# Extract x and y values for each metric
x = list(sae_data.keys())
metrics = ["disentangle", "isolate", "cause"]
y_values = {metric: [sae_data[key][metric] for key in x] for metric in metrics}

# Create a single plot
plt.figure(figsize=(10, 6))

# Plot each metric
for metric, values in y_values.items():
    plt.plot(x, values, marker="o", label=metric)

# Set plot attributes
plt.title("SAE Metrics")
plt.xlabel("Key")
plt.ylabel("Score")
plt.xscale("log")  # Set x-axis to logarithmic scale
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)

# Adjust layout and display the plot
plt.tight_layout()
plt.show()

# %%
# #@markdown Plotting

# import matplotlib.pyplot as plt
# import matplotlib

# plt.rcParams['figure.dpi'] = 100
# plt.rcParams['savefig.dpi'] = 100


# colors = [matplotlib.colors.to_hex(c) for c in plt.cm.tab20.colors]

# name_to_color = {
#     'SAE_RAND': 'gray',
#     'PCA': colors[6],
#     'SAE': colors[2],
#     'RLAP': colors[4],
#     'DBM': colors[1],
#     'MDBM': colors[0],
#     'DAS': colors[9],
#     'MDAS': colors[8],
# }

# name_to_marker = {
#     'SAE_RAND': 'o--',
#     'PCA': 'o--',
#     'SAE': 'o--',
#     'RLAP': '^--',
#     'DBM': 's--',
#     'MDBM': 's--',
#     'DAS': 's--',
#     'MDAS': 's--',
# }

# for n, x in method_to_data.items():
#   sorted_dim = sorted(x, key=lambda i: float(i[:-1]))
#   p = plt.plot([x[k][2] for k in sorted_dim],
#                [x[k][1] for k in sorted_dim], name_to_marker[n], label=n, markersize=10,
#                c=name_to_color[n])
#   for k in sorted(x, key=lambda s: x[s][0], reverse=True):
#     c = p[-1].get_color()
#     offset = (0, 0.05)
#     # Shift text boxes to avoid overlaps.
#     if n == 'SAE' and k == '3.8%':
#       offset = (0.05, -0.07)
#     plt.annotate(k, (x[k][2] - offset[0], x[k][1] + offset[1]), size=12,
#                  bbox=dict(boxstyle='round,pad=0.15', fc=c, ec='white', alpha=0.5))
# plt.scatter(1, 1, s=500, marker='*', color='gold', zorder=3)
# plt.annotate('GOAL', (1.0-0.18, 1.0 - 0.01), size=12)
# plt.gca().set_aspect('equal')
# plt.xlim(-0.1, 1.05)
# plt.ylim(-0.0, 1.1)
# plt.grid(alpha=0.3, linestyle='--')
# plt.legend(loc = 'lower left', prop={'size': 10})
# plt.xlabel('Cause Score', fontsize=12)
# _ = plt.ylabel('Isolate Score', fontsize=12)
