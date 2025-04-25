"""
Module for evaluating and measuring bias reduction through summary fusion and rewriting
using a RoBERTa+BiLSTM classifier and a fine-tuned LLaMA-2 LoRA model (CPU-only).
"""

import re
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from peft import PeftModel

# 1) CPU only (to match training)
device = torch.device("cpu")

# ----------------------------------------------
# 2) Load RoBERTa + BiLSTM bias classifier
# ----------------------------------------------
roberta_tokenizer = AutoTokenizer.from_pretrained("PlanTL-GOB-ES/roberta-base-bne")
roberta_model     = AutoModel.from_pretrained(
    "PlanTL-GOB-ES/roberta-base-bne",
    torch_dtype=torch.float32,
    device_map={"": "cpu"}
).to(device)

class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM classifier for text bias detection.

    Attributes
    ----------
    input_dim : int
        Dimensionality of input embeddings.
    hidden_dim : int
        Size of the hidden state in each LSTM direction.
    dropout : float
        Dropout probability within LSTM layers and after FC.
    """
    def __init__(self, input_dim, hidden_dim, dropout):
        """
        Initialize the BiLSTMClassifier.

        Parameters
        ----------
        input_dim : int
            Dimension of each input embedding vector.
        hidden_dim : int
            Number of hidden units per LSTM direction.
        dropout : float
            Dropout probability for regularization.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=3,
                            bidirectional=True, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.gelu = nn.GELU()

    def forward(self, x):
        """
        Forward pass: encode with BiLSTM, apply normalization, activation, and linear layers.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, input_dim).

        Returns
        -------
        torch.Tensor
            Output bias logits of shape (batch_size,).
        """
        lstm_out, _ = self.lstm(x)
        h_fw = lstm_out[:, -1, :self.hidden_dim]
        h_bw = lstm_out[:, -1, self.hidden_dim:]
        h = torch.cat([h_fw, h_bw], dim=1)
        h = self.layer_norm(h)
        h = self.dropout(self.gelu(self.fc1(h)))
        return self.fc2(h).squeeze()

classifier_model = BiLSTMClassifier(768, 256, 0.6).to(device)
classifier_model.load_state_dict(torch.load(
    "/Users/pablochamorro/Desktop/Coding/Thesis/best_bilstm_model.pth",
    map_location="cpu"
))
classifier_model.eval()


def split_sentences(text: str):
    """
    Split a text string into sentences using punctuation boundaries.

    Parameters
    ----------
    text : str
        The input text to split.

    Returns
    -------
    list[str]
        List of sentence substrings.
    """
    return re.split(r'(?<=[\.\?\!])\s+', text)


def get_embeddings(text: str, max_length: int = 512):
    """
    Compute mean-pooled RoBERTa embeddings for potentially long text by chunking.

    Parameters
    ----------
    text : str
        The text to embed.
    max_length : int, optional
        Maximum tokens per chunk (default: 512).

    Returns
    -------
    torch.Tensor
        Aggregated embedding tensor of shape (1, hidden_dim*2).
    """
    sentences = split_sentences(text)
    chunks, current = [], ""
    for s in sentences:
        cand = (current + " " + s).strip() if current else s
        ids  = roberta_tokenizer.encode(cand, add_special_tokens=True, truncation=False)
        if len(ids) <= max_length:
            current = cand
        else:
            if current:
                chunks.append(current)
            current = s
    if current:
        chunks.append(current)

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


def predict_bias(text: str):
    """
    Predict a scaled bias score for a given text using the trained classifier.

    Parameters
    ----------
    text : str
        Input text to evaluate.

    Returns
    -------
    float
        Scaled bias value between 0 and 1.
    """
    embedding = get_embeddings(text).unsqueeze(0).to(device)
    with torch.no_grad():
        bias = torch.sigmoid(classifier_model(embedding)).item()
    k = 3
    return abs(np.tanh((bias - 0.5) * k))

# ----------------------------------------------
# 3) Load fine-tuned LLaMA-2 + LoRA (CPU)
# ----------------------------------------------
fine_tuned = "/Users/pablochamorro/Desktop/Coding/Thesis/fine_tuned_gen_model2_100"
llama_tokenizer = AutoTokenizer.from_pretrained(fine_tuned, padding_side="left")
llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_tokenizer.bos_token = llama_tokenizer.eos_token

base = AutoModelForCausalLM.from_pretrained(
    fine_tuned,
    torch_dtype=torch.float32
)
llama_model = PeftModel.from_pretrained(
    base,
    fine_tuned,
    torch_dtype=torch.float32
)
llama_model.to(device)
llama_model.eval()

# ----------------------------------------------
# 4) Prompt templates
# ----------------------------------------------
def extract_prompt(text: str) -> str:
    """
    Build a prompt for extracting relevant facts and figures from text.

    Parameters
    ----------
    text : str
        Original article text to summarize.

    Returns
    -------
    str
        Formatted extraction prompt.
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
    Build a prompt for fusing two summaries into one neutral article.

    Parameters
    ----------
    sum_m : str
        Summary generated from El Mundo text.
    sum_p : str
        Summary generated from El País text.

    Returns
    -------
    str
        Formatted fusion prompt.
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
    Generate text from the fine-tuned LLaMA model given a prompt.

    Parameters
    ----------
    prompt : str
        The instruction prompt.
    max_new : int
        Maximum number of tokens to generate.

    Returns
    -------
    str
        The generated and post-processed text.
    """
    inputs = llama_tokenizer(
        prompt,
        return_tensors="pt",
        padding=True, truncation=True, max_length=2048
    ).to(device)
    inputs.pop("token_type_ids", None)
    in_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = llama_model.generate(
            **inputs,
            max_new_tokens      = max_new,
            do_sample           = False,
            forced_bos_token_id = llama_tokenizer.bos_token_id,
            eos_token_id        = llama_tokenizer.eos_token_id,
            pad_token_id        = llama_tokenizer.pad_token_id,
            early_stopping      = True,
        )
    gen = out[0][in_len:]
    text = llama_tokenizer.decode(gen, skip_special_tokens=True).strip()
    if text and not text[0].isupper():
        text = text[0].upper() + text[1:]
    return text

# ----------------------------------------------
# 5) Load test set
# ----------------------------------------------
test_df = pd.read_csv("/Users/pablochamorro/Desktop/Coding/Thesis/structured_test_pairs.csv")

def get_pair(df, date):
    """
    Retrieve paired texts (El Mundo and El País) for a given date from DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'date', 'source', and 'bodytext' columns.
    date : str
        The target date to filter by.

    Returns
    -------
    tuple[str, str] or (None, None)
        Returns (mundo_text, pais_text) if both present, else (None, None).
    """
    sub   = df[df.date == date]
    mundo = sub[sub.source.str.lower()=="el mundo"].bodytext.values
    pais  = sub[sub.source.str.lower()=="el país"].bodytext.values
    if len(mundo) and len(pais):
        return mundo[0], pais[0]
    return None, None

# ----------------------------------------------
# 6) Evaluate on test dates and record results
# ----------------------------------------------
results = []
original_biases  = []
rewritten_biases = []

for date in test_df.date.unique():
    pair = get_pair(test_df, date)
    if not pair:
        continue
    m, p = pair

    ob = 0.5 * (predict_bias(m) + predict_bias(p))
    original_biases.append(ob)

    # 1) extract
    sum_m = run_chat(extract_prompt(m), max_new=256)
    sum_p = run_chat(extract_prompt(p), max_new=256)

    # 2) fuse
    fp = fusion_prompt(sum_m, sum_p)
    length_in = llama_tokenizer(fp, return_tensors="pt", truncation=True, max_length=2048)["input_ids"].shape[1]
    max_new = int(length_in * 0.9)
    fused = run_chat(fp, max_new=max_new)

    rb = predict_bias(fused)
    rewritten_biases.append(rb)

    results.append({
        "date": date,
        "bias_original": ob,
        "summary_mundo": sum_m,
        "summary_pais": sum_p,
        "fused_article": fused,
        "bias_rewritten": rb
    })

    print(f"Fecha: {date} | Orig bias: {ob:.4f} | Rew bias: {rb:.4f}")
    print("Rewritten (fused) article:")
    print(fused)
    print("-" * 80)

# ----------------------------------------------
# 7) Save to CSV
# ----------------------------------------------
out_df = pd.DataFrame(results)
out_df.to_csv("/Users/pablochamorro/Desktop/Coding/Thesis/test_results_final.csv", index=False)
print("Saved results to test_results_final.csv")

# ----------------------------------------------
# 8) Compute overall averages
# ----------------------------------------------
overall_original_bias  = np.mean(original_biases)  if original_biases  else 0
overall_rewritten_bias = np.mean(rewritten_biases) if rewritten_biases else 0

print("\nPromedio de sesgo en los textos originales (combinados por fecha): {:.4f}".format(overall_original_bias))
print("Promedio de sesgo en los textos reescritos: {:.4f}".format(overall_rewritten_bias))
