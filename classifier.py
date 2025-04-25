import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
import spacy


# Check for GPU availability
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
nlp = spacy.load("es_core_news_sm")

# Load Dataset
df = pd.read_csv("/Users/pablochamorro/Desktop/Coding/Thesis/most_similar_articles_train_classification.csv")

# Assign labels with soft values
df['el_mundo_label'] = 1  # El Mundo → Right-wing
df['el_pais_label'] = 0   # El País → Left-wing

# Load Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained("PlanTL-GOB-ES/roberta-base-bne")
model_roberta = AutoModel.from_pretrained("PlanTL-GOB-ES/roberta-base-bne").to(device)


def preprocess_text(text):
    """
    Lowercase, lemmatize, and remove stopwords and punctuation from a text string.

    Parameters
    ----------
    text : str
        Input text to be normalized.

    Returns
    -------
    str
        Processed text containing space-joined lemmas without stopwords or punctuation.
    """
    doc = nlp(text.lower())  # Normalize to lowercase
    processed_text = " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])
    return processed_text


def get_embeddings(text, max_length=512, overlap=256):
    """
    Generate a mean-pooled embedding for long text by chunking through a tokenizer and model.

    Parameters
    ----------
    text : str
        The input text to embed.
    max_length : int, optional
        Maximum token length per chunk including overlap (default is 512).
    overlap : int, optional
        Number of tokens to overlap between chunks (default is 256).

    Returns
    -------
    torch.Tensor
        A single tensor of shape (1, hidden_size) representing the aggregated embedding.
    """
    tokens = tokenizer.encode(text, add_special_tokens=True, truncation=False)
    chunk_size = max_length - overlap
    chunks = [tokens[i: i + max_length] for i in range(0, len(tokens), chunk_size)]

    embeddings = []
    for chunk in chunks:
        inputs = tokenizer.batch_encode_plus(
            [tokenizer.decode(chunk)],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = model_roberta(**inputs)

        embeddings.append(outputs.last_hidden_state.mean(dim=1))

    return torch.mean(torch.stack(embeddings), dim=0)


def normalize(tensor):
    """
    Normalize a tensor to zero mean and unit variance.

    Parameters
    ----------
    tensor : torch.Tensor
        The tensor to normalize.

    Returns
    -------
    torch.Tensor
        The normalized tensor.
    """
    return (tensor - tensor.mean()) / (tensor.std() + 1e-6)


# Generate embeddings for El Mundo with progress tracking
print("Generating embeddings for El Mundo articles...")
df['el_mundo_embedding'] = [
    normalize(get_embeddings(text).cpu()) if pd.notna(text) else None
    for text in tqdm(df['El_Mundo_bodytext'], desc="El Mundo Progress")
]

# Generate embeddings for El País with progress tracking
print("Generating embeddings for El País articles...")
df['el_pais_embedding'] = [
    normalize(get_embeddings(text).cpu()) if pd.notna(text) else None
    for text in tqdm(df['El_Pais_bodytext'], desc="El País Progress")
]

# Drop rows with missing embeddings
df = df.dropna(subset=['el_mundo_embedding', 'el_pais_embedding'])

# Convert embeddings to tensors
X = torch.cat(list(df['el_mundo_embedding']) + list(df['el_pais_embedding']))
y = torch.tensor(
    [1] * len(df['el_mundo_embedding']) + [0] * len(df['el_pais_embedding']),
    dtype=torch.float32
)

# Move tensors to device
X, y = X.to(device), y.to(device)

# Create dataset and dataloader
dataset = TensorDataset(X, y)
train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)


class BiLSTMClassifier(nn.Module):
    """
    A bidirectional LSTM classifier for sequence embeddings.

    Architecture:
      - 3-layer bidirectional LSTM
      - Layer normalization
      - GELU activation
      - Dropout
      - Two fully connected layers
    """

    def __init__(self, input_dim, hidden_dim, dropout):
        """
        Initialize the BiLSTMClassifier.

        Parameters
        ----------
        input_dim : int
            Dimensionality of input embeddings per time step.
        hidden_dim : int
            Number of hidden units in the LSTM layers.
        dropout : float
            Dropout probability between layers.
        """
        super(BiLSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=3,
            bidirectional=True,
            batch_first=True,
            dropout=dropout
        )
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.gelu = nn.GELU()

    def forward(self, x):
        """
        Perform a forward pass through the network.

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
        print(f"LSTM output shape: {lstm_out.shape}")  # Debugging line

        if lstm_out.dim() == 3:
            hidden = torch.cat(
                (lstm_out[:, -1, :self.hidden_dim], lstm_out[:, -1, self.hidden_dim:]),
                dim=1
            )
        else:
            hidden = lstm_out  # Handles case where seq dim is missing

        hidden = self.layer_norm(hidden)
        hidden = self.dropout(self.gelu(self.fc1(hidden)))
        output = self.fc2(hidden).squeeze()
        return output


print("TRAINING!")


def train_model(train_dataloader, model, optimizer, criterion, epochs):
    """
    Train the model for a given number of epochs.

    Parameters
    ----------
    train_dataloader : DataLoader
        DataLoader providing (input, label) batches.
    model : nn.Module
        The model to train.
    optimizer : torch.optim.Optimizer
        Optimizer for model parameters.
    criterion : loss function
        Loss function to minimize.
    epochs : int
        Number of training epochs.

    Side Effects
    ------------
    Prints epoch loss after each epoch.
    """
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for inputs, labels in train_dataloader:
            optimizer.zero_grad()
            output = model(inputs.unsqueeze(1))  # Add sequence length dimension
            loss = criterion(output, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")


# Define hyperparameter grid
param_grid = {
    'hidden_dim': [64, 128, 256],
    'dropout': [0.2, 0.4, 0.6],
    'lr': [0.0001, 0.0005, 0.001, 0.002]
}

best_loss = float('inf')
best_params = None
best_model = None

# Grid search for best hyperparameters
for params in ParameterGrid(param_grid):
    print(f"Testing params: {params}")
    model = BiLSTMClassifier(
        input_dim=768,
        hidden_dim=params['hidden_dim'],
        dropout=params['dropout']
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    train_model(train_dataloader, model, optimizer, criterion, epochs=15)

    # Evaluate final loss
    with torch.no_grad():
        total_loss = 0
        for inputs, labels in train_dataloader:
            output = model(inputs.unsqueeze(1))
            loss = criterion(output, labels)
            total_loss += loss.item()

    if total_loss < best_loss:
        best_loss = total_loss
        best_params = params
        best_model = model

# Save best model and parameters
print(f"Best parameters: {best_params}, Loss: {best_loss:.4f}")
torch.save(best_model.state_dict(), "/Users/pablochamorro/Desktop/Coding/Thesis/best_bilstm_model.pth")

# Generate bias probabilities with best model
with torch.no_grad():
    el_mundo_tensors = torch.cat(
        [torch.tensor(embed, dtype=torch.float32) for embed in df['el_mundo_embedding']]
    ).to(device)
    el_pais_tensors = torch.cat(
        [torch.tensor(embed, dtype=torch.float32) for embed in df['el_pais_embedding']]
    ).to(device)

    df['el_mundo_bias_prob'] = torch.sigmoid(best_model(el_mundo_tensors)).cpu().numpy()
    df['el_pais_bias_prob'] = torch.sigmoid(best_model(el_pais_tensors)).cpu().numpy()

# Print probabilities
print(df[['el_mundo_bias_prob', 'el_pais_bias_prob']].head())
