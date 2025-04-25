# -*- coding: utf-8 -*-
"""
Module for bias classification and reinforcement learning fine-tuning
using RoBERTa, BiLSTM, and meta-llama LLaMA-2 with LoRA adapter.
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from peft import get_peft_model, LoraConfig, TaskType

# --------------------------------------------
#  Settings: CPU-only, suppress warnings
# --------------------------------------------
logging.set_verbosity_error()
device = torch.device("cpu")

# --------------------------------------------
#  Bias classifier (RoBERTa + BiLSTM)
# --------------------------------------------
roberta_tokenizer = AutoTokenizer.from_pretrained("PlanTL-GOB-ES/roberta-base-bne")
roberta_model     = AutoModel.from_pretrained(
    "PlanTL-GOB-ES/roberta-base-bne", torch_dtype=torch.float32, device_map={"": "cpu"}
).to(device)

class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM classifier for bias detection.

    Attributes
    ----------
    input_dim : int
        Dimensionality of input embeddings.
    hidden_dim : int
        Number of hidden units in each LSTM direction.
    dropout : float
        Dropout probability between layers.
    """
    def __init__(self, input_dim, hidden_dim, dropout):
        """
        Initialize the BiLSTMClassifier.

        Parameters
        ----------
        input_dim : int
            Size of each input embedding vector.
        hidden_dim : int
            Size of the hidden state in each LSTM layer (per direction).
        dropout : float
            Dropout probability applied within the LSTM and on the FC layer.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            num_layers=3, bidirectional=True,
                            batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.gelu = nn.GELU()

    def forward(self, x):
        """
        Forward pass through BiLSTM and classification layers.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, input_dim).

        Returns
        -------
        torch.Tensor
            Output logits tensor of shape (batch_size,).
        """
        lstm_out, _ = self.lstm(x)
        h_fw = lstm_out[:, -1, :self.hidden_dim]
        h_bw = lstm_out[:, -1, self.hidden_dim:]
        h    = torch.cat([h_fw, h_bw], dim=1)
        h    = self.layer_norm(h)
        h    = self.dropout(self.gelu(self.fc1(h)))
        return self.fc2(h).squeeze()

# load classifier
classifier = BiLSTMClassifier(768, 256, 0.6).to(device)
classifier.load_state_dict(torch.load(
    "/Users/pablochamorro/Desktop/Coding/Thesis/best_bilstm_model.pth",
    map_location="cpu"
))
classifier.eval()

def split_sentences(text: str):
    """
    Split a text string into sentences based on punctuation.

    Parameters
    ----------
    text : str
        The input text to split.

    Returns
    -------
    list of str
        List of sentence strings.
    """
    return re.split(r'(?<=[\.\?\!])\s+', text)


def get_embeddings(text: str, max_length: int = 512):
    """
    Compute mean-pooled RoBERTa embeddings for a potentially long text by chunking.

    Sentences are concatenated until token limit, then embedded in chunks.

    Parameters
    ----------
    text : str
        The input text to embed.
    max_length : int, optional
        Maximum number of tokens per chunk (default=512).

    Returns
    -------
    torch.Tensor
        A tensor of shape (1, hidden_size) representing the aggregated embedding.
    """
    sentences = split_sentences(text)
    chunks, current = [], ""
    for s in sentences:
        cand = (current + " " + s).strip() if current else s
        ids  = roberta_tokenizer.encode(cand, add_special_tokens=True, truncation=False)
        if len(ids) <= max_length:
            current = cand
        else:
            if current: chunks.append(current)
            current = s
    if current: chunks.append(current)

    embeds = []
    for chunk in chunks:
        inputs = roberta_tokenizer(
            chunk,
            return_tensors="pt",
            padding=True, truncation=True,
            max_length=max_length
        ).to(device)
        with torch.no_grad():
            out = roberta_model(**inputs)
        embeds.append(out.last_hidden_state.mean(dim=1))
    return torch.mean(torch.stack(embeds), dim=0)


def predict_bias(text):
    """
    Predict a bias score for a piece of text using the trained BiLSTM classifier.

    Bias is scaled via a tanh-based adjustment to [0,1].

    Parameters
    ----------
    text : str
        The input text to evaluate.

    Returns
    -------
    float
        Absolute bias score between 0 and 1.
    """
    embedding = get_embeddings(text).unsqueeze(0).to(device)
    with torch.no_grad():
        bias = torch.sigmoid(classifier(embedding)).item()
    k = 3
    return abs(np.tanh((bias - 0.5) * k))

# --------------------------------------------
#  Load LLaMA-2 + LoRA adapter (CPU)
# --------------------------------------------
base_name = "meta-llama/Llama-2-7b-chat-hf"
tok = AutoTokenizer.from_pretrained(base_name, padding_side="left")
tok.pad_token = tok.eos_token
tok.bos_token = tok.eos_token

model_base = AutoModelForCausalLM.from_pretrained(
    base_name, torch_dtype=torch.float32, device_map={"": "cpu"}
)
lora_cfg = LoraConfig(
    r=8, lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model_base, lora_cfg).to(device)
optimizer = optim.AdamW(model.parameters(), lr=2e-5)

# --------------------------------------------
#  Prompt builders
# --------------------------------------------
def extract_prompt(text: str) -> str:
    """
    Build a prompt to extract key facts and figures from an article.

    Parameters
    ----------
    text : str
        Original article text.

    Returns
    -------
    str
        System prompt wrapping the input text for extraction.
    """
    return (
        "<s>[INST] <<SYS>>\n"
        "Eres un periodista. Extrae los hechos y cifras más relevantes de este texto, "
        "de forma clara y concisa y completamente en español.\n"
        "<</SYS>>\n\n"
        f"{text}\n\n"
        "[/INST]\n"
    )

def fusion_prompt(sum_m: str, sum_p: str) -> str:
    """
    Build a prompt to fuse two summaries into one neutral article.

    Parameters
    ----------
    sum_m : str
        Summary from El Mundo.
    sum_p : str
        Summary from El País.

    Returns
    -------
    str
        System prompt for fusing the two summaries.
    """
    return (
        "<s>[INST] <<SYS>>\n"
        "Eres un periodista que fusiona dos resúmenes en un solo artículo completamente neutral y objetivo. "
        "Mantén todos los hechos y cifras relevantes, incluidas fechas, porcentajes y nombres de partidos. "
        "El artículo debe estar en español, informativo en un tono neutral y aproximadamente un 10% más corto\n"
        "que la longitud combinada de los dos resúmenes.\n"
        "<</SYS>>\n\n"
        f"Resumen El Mundo:\n{sum_m}\n\n"
        f"Resumen El País:\n{sum_p}\n\n"
        "[/INST]\n"
    )

def run_chat(prompt: str, max_new: int) -> str:
    """
    Generate text from the LLaMA model given a prompt.

    Parameters
    ----------
    prompt : str
        The input prompt to the model.
    max_new : int
        Maximum number of new tokens to generate.

    Returns
    -------
    str
        Generated text, capitalized at start.
    """
    inputs = tok(
        prompt,
        return_tensors="pt",
        padding=True, truncation=True,
        max_length=2048
    ).to(device)
    inputs.pop("token_type_ids", None)
    in_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens      = max_new,
            do_sample           = False,
            forced_bos_token_id = tok.bos_token_id,
            eos_token_id        = tok.eos_token_id,
            pad_token_id        = tok.pad_token_id,
            early_stopping      = True,
        )
    gen = out[0][in_len:]
    text = tok.decode(gen, skip_special_tokens=True).strip()
    if text and not text[0].isupper():
        text = text[0].upper() + text[1:]
    return text

# --------------------------------------------
#  Data loading
# --------------------------------------------
train_df = pd.read_csv("/Users/pablochamorro/Desktop/Coding/Thesis/structured_train_pairs.csv")

def get_pair(date: str):
    """
    Retrieve the article pair (El Mundo, El País) for a given date.

    Parameters
    ----------
    date : str
        Date string to filter the dataset.

    Returns
    -------
    tuple[str, str] or (None, None)
        Returns (mundo_text, pais_text) or (None, None) if missing.
    """
    sub   = train_df[train_df.date == date]
    mundo = sub[sub.source.str.lower()=="el mundo"].bodytext.values
    pais  = sub[sub.source.str.lower()=="el país"].bodytext.values
    if len(mundo) and len(pais):
        return mundo[0], pais[0]
    return None, None

valid_dates = [d for d in train_df.date.unique() if get_pair(d)[0] is not None]

# --------------------------------------------
#  RL training step
# --------------------------------------------
reward_baseline = 0.0
baseline_w      = 0.9
results         = []

def reinforce_step(mundo: str, pais: str, orig_bias: float):
    """
    Perform one reinforcement learning step combining summaries and computing policy gradient.

    Steps:
      1) Extract summaries
      2) Fuse summaries
      3) Generate fused article
      4) Compute reward based on bias reduction
      5) Update model via policy gradient

    Parameters
    ----------
    mundo : str
        Original El Mundo article text.
    pais : str
        Original El País article text.
    orig_bias : float
        Baseline bias score of original articles.

    Returns
    -------
    tuple
        (reward, bias_diff, policy_loss, generated_text)
    """
    global reward_baseline

    # 1) extract summaries
    sum_m = run_chat(extract_prompt(mundo), max_new=256)
    sum_p = run_chat(extract_prompt(pais),  max_new=256)

    # 2) fuse summaries
    fusion_p = fusion_prompt(sum_m, sum_p)
    # compute roughly 90% of prompt length for final
    tokens_in = tok(fusion_p, return_tensors="pt", truncation=True, max_length=2048)["input_ids"].shape[1]
    max_new   = int(tokens_in * 0.9)

    model.eval()
    with torch.no_grad():
        out_ids = model.generate(
            **tok(fusion_p, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device),
            max_new_tokens      = max_new,
            do_sample           = True,
            temperature         = 0.4,
            top_p               = 0.9,
            repetition_penalty  = 1.2,
            forced_bos_token_id = tok.bos_token_id,
            eos_token_id        = tok.eos_token_id,
            pad_token_id        = tok.pad_token_id,
            early_stopping      = True,
        )
    model.train()

    # get generated article
    gen = out_ids[0][tokens_in:]
    gen_text = tok.decode(gen, skip_special_tokens=True).strip()
    if gen_text and not gen_text[0].isupper():
        gen_text = gen_text[0].upper() + gen_text[1:]
    if len(gen_text.split()) < 20:
        gen_text = "<Texto demasiado corto>"

    # compute reward
    new_bias = predict_bias(gen_text)
    reward   = orig_bias - new_bias
    reward_baseline = baseline_w*reward_baseline + (1-baseline_w)*reward
    advantage = reward - reward_baseline

    # policy gradient
    logits = model(tok(fusion_p, return_tensors="pt").to(device)["input_ids"]).logits[:, :-1, :]
    logp   = F.log_softmax(logits, dim=-1)
    sel    = logp[0, -len(gen):, gen]
    policy_loss = -advantage * sel.mean()

    optimizer.zero_grad()
    policy_loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return reward, orig_bias-new_bias, policy_loss.item(), gen_text

# --------------------------------------------
#  Training loop
# --------------------------------------------
print("⚡️ Starting RL fine-tuning…")
for epoch in range(1):
    print(f"Epoch {epoch+1}")
    for date in tqdm(valid_dates):
        m, p = get_pair(date)
        orig = 0.5*(predict_bias(m)+predict_bias(p))
        rwd, bias_diff, loss, out = reinforce_step(m, p, orig)
        results.append({
            "date": date,
            "rewritten": out,
            "original_bias": orig,
            "new_bias": orig-bias_diff,
            "reward": rwd,
            "loss": loss
        })
        print(f"{date} | R: {rwd:.3f} | Δbias: {bias_diff:.3f} | L: {loss:.3f}")

# --------------------------------------------
#  Save model & results
# --------------------------------------------
save_path = "/Users/pablochamorro/Desktop/Coding/Thesis/fine_tuned_gen_model_pairs"
model.save_pretrained(save_path)
tok.save_pretrained(save_path)
pd.DataFrame(results).to_csv(
    "/Users/pablochamorro/Desktop/Coding/Thesis/test_results_pairs.csv", index=False
)
print("Fine-tuning complete and saved.")
