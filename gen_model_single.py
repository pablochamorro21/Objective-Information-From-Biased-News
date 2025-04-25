import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import numpy as np
import pandas as pd
from tqdm import tqdm
from peft import get_peft_model, LoraConfig, TaskType

# Configuración del dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------
# MODELO DE SESGO (RoBERTa y clasificador BiLSTM)
# ----------------------------------------------
roberta_tokenizer = AutoTokenizer.from_pretrained("PlanTL-GOB-ES/roberta-base-bne")
roberta_model = AutoModel.from_pretrained("PlanTL-GOB-ES/roberta-base-bne").to(device)

# Cargar el dataset; se asume que cada fila es un artículo (El Mundo o El País).
train_dataset = pd.read_csv("/Users/pablochamorro/Desktop/Coding/Thesis/structured_train_pairs.csv")

class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM classifier for sequence classification tasks.
    """
    def __init__(self, input_dim, hidden_dim, dropout):
        """
        Initialize the BiLSTMClassifier.

        Parameters
        ----------
        input_dim : int
            Dimensionality of the input embeddings.
        hidden_dim : int
            Number of hidden units in each LSTM layer.
        dropout : float
            Dropout probability applied between layers.
        """
        super(BiLSTMClassifier, self).__init__()
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
        Forward pass of the BiLSTM classifier.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, input_dim).

        Returns
        -------
        torch.Tensor
            Output logits of shape (batch_size,).
        """
        lstm_out, _ = self.lstm(x)
        hidden = torch.cat((lstm_out[:, -1, :self.hidden_dim],
                            lstm_out[:, -1, self.hidden_dim:]), dim=1)
        hidden = self.layer_norm(hidden)
        hidden = self.dropout(self.gelu(self.fc1(hidden)))
        output = self.fc2(hidden).squeeze()
        return output

params = {'dropout': 0.6, 'hidden_dim': 256, 'lr': 2e-5}
classifier_model = BiLSTMClassifier(768, params['hidden_dim'], params['dropout']).to(device)
classifier_model.load_state_dict(torch.load("/Users/pablochamorro/Desktop/Coding/Thesis/best_bilstm_model.pth"))
classifier_model.eval()

def get_embeddings(text, max_length=512, overlap=256):
    """
    Generate a mean-pooled embedding for possibly long text by chunking inputs.

    Parameters
    ----------
    text : str
        The input text to embed.
    max_length : int, optional
        Maximum token length per chunk including overlap (default is 512).
    overlap : int, optional
        Number of tokens to overlap between consecutive chunks (default is 256).

    Returns
    -------
    torch.Tensor
        Aggregated embedding tensor of shape (1, hidden_size).
    """
    tokens = roberta_tokenizer.encode(text, add_special_tokens=True, truncation=False)
    chunk_size = max_length - overlap
    chunks = [tokens[i:i+max_length] for i in range(0, len(tokens), chunk_size)]
    embeddings = []
    for chunk in chunks:
        inputs_chunk = roberta_tokenizer.batch_encode_plus(
            [roberta_tokenizer.decode(chunk)],
            return_tensors="pt", padding=True, truncation=True, max_length=max_length
        )
        inputs_chunk = {k: v.to(device) for k, v in inputs_chunk.items()}
        with torch.no_grad():
            outputs = roberta_model(**inputs_chunk)
        embeddings.append(outputs.last_hidden_state.mean(dim=1))
    return torch.mean(torch.stack(embeddings), dim=0)

def predict_bias(text):
    """
    Predict bias score of a given text using the pre-trained BiLSTM classifier.

    Parameters
    ----------
    text : str
        Input text to evaluate bias.

    Returns
    -------
    float
        Absolute bias score scaled between 0 and 1.
    """
    embedding = get_embeddings(text).unsqueeze(0).to(device)
    with torch.no_grad():
        bias = torch.sigmoid(classifier_model(embedding)).item()
    k = 3
    return abs(np.tanh((bias - 0.5) * k))

# ----------------------------------------------
# MODELO GENERATIVO
# ----------------------------------------------
model_name_gen = "meta-llama/Llama-2-7b-chat-hf"
llama_tokenizer = AutoTokenizer.from_pretrained(model_name_gen)
llama_tokenizer.pad_token = llama_tokenizer.eos_token
base_model = AutoModelForCausalLM.from_pretrained(
    model_name_gen, torch_dtype=torch.float32, device_map={"": device}
)
lora_config = LoraConfig(
    r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1, bias="none", task_type=TaskType.CAUSAL_LM
)
llama_model = get_peft_model(base_model, lora_config)
llama_model.train()
optimizer = optim.AdamW(llama_model.parameters(), lr=params['lr'])

def format_prompt(text):
    """
    Format the instruction prompt for neutral news rewriting.

    Parameters
    ----------
    text : str
        Original article text to be rewritten.

    Returns
    -------
    str
        Prompt padded with system instructions and original text.
    """
    return (
        "<s>[INST] <<SYS>>\n"
        "Eres un asistente que reescribe textos de noticias en español de manera neutral y objetiva, eliminando sesgos y opiniones. "
        "Reescribe el siguiente artículo conservando todos los hechos relevantes, pero hazlo aproximadamente un 10% más corto que el original. "
        "El resultado debe estar completamente en español.\n"
        "<</SYS>>\n\n"
        "Texto original:\n"
        f"{text}\n\n"
        "Texto reescrito:\n[/INST]"
    )

def get_article(row):
    """
    Extract the article text from a DataFrame row.

    Parameters
    ----------
    row : pandas.Series
        A single row of the dataset containing 'bodytext'.

    Returns
    -------
    str
        The content of the 'bodytext' field.
    """
    return row["bodytext"]

# Contar muestras válidas (filas con contenido en 'bodytext')
valid_indices = []
for idx, row in train_dataset.iterrows():
    if pd.notna(row["bodytext"]) and len(row["bodytext"].strip()) > 0:
        valid_indices.append(idx)
print(f"Found {len(valid_indices)} valid samples in the dataset.")

# ----------------------------------------------
# ENTRENAMIENTO POR REFUERZO
# ----------------------------------------------
results = []
reward_history = []
bias_diff_history = []
loss_history = []
reward_baseline = 0.0
baseline_weight = 0.9

def reinforce_step(article_text, original_bias):
    """
    Perform one reinforcement learning step on a single article.

    Parameters
    ----------
    article_text : str
        Original article text.
    original_bias : float
        Bias probability of the original article.

    Returns
    -------
    tuple
        reward (float): Reward for bias reduction.
        bias_diff (float): Change in bias.
        policy_loss (float): Policy gradient loss value.
        article_text (str): Original text input.
        generated_text (str): Rewritten text generated.
    """
    global reward_baseline
    prompt = format_prompt(article_text)
    inputs = llama_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
    if "token_type_ids" in inputs:
        inputs.pop("token_type_ids")
    
    llama_model.eval()
    with torch.no_grad():
        outputs = llama_model.generate(
            **inputs,
            max_new_tokens=1000,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.1,
            pad_token_id=llama_tokenizer.pad_token_id
        )
    llama_model.train()
    
    prompt_length = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][prompt_length:]
    generated_text = llama_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    if not generated_text or len(generated_text.split()) < 10:
        print("⚠️ Empty or too short output from model. Skipping.")
        return 0.0, 0.0, 0.0, article_text, "[Empty Output]"
    
    if generated_text.strip() == prompt.strip():
        print("⚠️ Generated text is identical to the prompt.")
    
    new_bias = predict_bias(generated_text)
    reward = original_bias - new_bias   # Positive if bias reduced.
    bias_diff = reward
    reward_baseline = baseline_weight * reward_baseline + (1 - baseline_weight) * reward
    advantage = reward - reward_baseline

    model_inputs = outputs[:, :-1]
    logits = llama_model(model_inputs).logits.squeeze(0)
    log_probs = F.log_softmax(logits, dim=-1)[-len(generated_ids):]
    selected_log_probs = log_probs[torch.arange(len(generated_ids)), generated_ids]
    
    policy_loss = -advantage * selected_log_probs.mean()
    
    optimizer.zero_grad()
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(llama_model.parameters(), 1.0)
    optimizer.step()
    
    return reward, bias_diff, policy_loss.item(), article_text, generated_text

print("\nStarting fine-tuning...\n")
num_epochs = 5
for epoch in range(num_epochs):
    print(f"\n=== Epoch {epoch + 1} ===")
    for i in tqdm(valid_indices, desc="Training samples"):
        row = train_dataset.loc[i]
        article_text = get_article(row)
        if article_text is None or len(article_text.strip()) == 0:
            continue
        original_bias = row["bias_prob"]
        reward, bias_diff, loss, original_text, rewritten = reinforce_step(article_text, original_bias)
        reward_history.append(reward)
        bias_diff_history.append(bias_diff)
        loss_history.append(loss)
        print(f"\nSample {i} | Reward: {reward:.4f} | Bias Δ: {bias_diff:.4f} | Loss: {loss:.4f}")
        print(f"Texto original (primeros 200 caracteres): {original_text[:200]}...")
        print(f"Texto reescrito (primeros 200 caracteres): {rewritten[:200]}...")
        results.append({
            "date": row["date"],
            "source": row["source"],
            "original_text": article_text,
            "rewritten": rewritten,
            "reward": reward,
            "bias_diff": bias_diff,
            "loss": loss
        })

print("\nFine-tuning complete.")

# Guardar el modelo y tokenizador fine-tuneado.
save_path = "/Users/pablochamorro/Desktop/Coding/Thesis/fine_tuned_gen_model"
llama_model.save_pretrained(save_path)
llama_tokenizer.save_pretrained(save_path)
print(f"Model and tokenizer saved to {save_path}")

# Guardar los resultados en un CSV.
results_df = pd.DataFrame(results)
results_csv_path = "/Users/pablochamorro/Desktop/Coding/Thesis/test_results_gen.csv"
results_df.to_csv(results_csv_path, index=False)
print(f"Results saved to {results_csv_path}")
