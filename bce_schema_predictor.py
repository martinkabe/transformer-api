import json
import torch
import torch.nn as nn
import torch.nn.functional as F


with open("training_data.json") as f:
    training_data = json.load(f)


training_data = [
    (["[CLS]"] + prompt, columns)
    for prompt, columns in training_data
]

all_tokens = set()
all_columns = set()
for prompt, columns in training_data:
    all_tokens.update(prompt)
    all_tokens.update(columns)
    all_columns.update(columns)
vocab = sorted(all_tokens)
columns_vocab = sorted(all_columns)

token_to_idx = {token: idx for idx, token in enumerate(vocab)}
idx_to_token = {idx: token for token, idx in token_to_idx.items()}
column_to_idx = {col: i for i, col in enumerate(columns_vocab)}
idx_to_column = {i: col for col, i in column_to_idx.items()}
vocab_size = len(vocab)
num_columns = len(columns_vocab)


def encode(tokens):
    return [token_to_idx[t] for t in tokens]

def encode_columns(cols):
    vec = torch.zeros(num_columns)
    for col in cols:
        if col in column_to_idx:
            vec[column_to_idx[col]] = 1.0
    return vec


def predict_columns(model, prompt_tokens, threshold=0.5):
    x = torch.tensor([encode(["[CLS]"] + prompt_tokens)], dtype=torch.long)
    logits = model(x)
    probs = torch.sigmoid(logits).squeeze()
    pred_indices = (probs > threshold).nonzero(as_tuple=True)[0].tolist()
    return [idx_to_column[i] for i in pred_indices]


class SchemaPredictor(nn.Module):
    def __init__(self, vocab_size, num_labels, embed_dim=64, num_heads=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, num_labels)

    def forward(self, x):
        x = self.embedding(x)
        attn_output, _ = self.attn(x, x, x)
        x = self.ln(x + attn_output)
        cls_token = x[:, 0, :]  # use [CLS] token
        return self.output_proj(cls_token)