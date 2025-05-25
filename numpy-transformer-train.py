
import numpy as np
import random
import json
from faker import Faker
from pprint import pprint

# Load training data
with open("training_data.json") as f:
    training_data = json.load(f)

# Vocabulary
all_tokens = set()
for prompt, target in training_data:
    all_tokens.update(prompt)
    all_tokens.update(target)
vocab = sorted(all_tokens)
token_to_idx = {t: i for i, t in enumerate(vocab)}
idx_to_token = {i: t for t, i in token_to_idx.items()}
vocab_size = len(vocab)

embed_dim = 32
num_heads = 2
lr = 0.01

# Weights
W_embed = np.random.randn(vocab_size, embed_dim)
W_proj = np.random.randn(embed_dim, vocab_size)
W_q = np.random.randn(embed_dim, embed_dim)
W_k = np.random.randn(embed_dim, embed_dim)
W_v = np.random.randn(embed_dim, embed_dim)

def encode(tokens):
    return [token_to_idx[t] for t in tokens]

def decode(indices):
    return [idx_to_token[i] for i in indices]

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def layer_norm(x):
    return (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-6)

def train():
    global W_embed, W_proj, W_q, W_k, W_v
    for epoch in range(400):
        total_loss = 0
        for prompt, target in training_data:
            idxs = encode(prompt)
            x = np.stack([W_embed[i] for i in idxs])[np.newaxis, :, :]
            B, T, D = x.shape
            head_dim = D // num_heads

            def split_heads(v): return v.reshape(B, T, num_heads, head_dim).transpose(0, 2, 1, 3)
            def merge_heads(v): return v.transpose(0, 2, 1, 3).reshape(B, T, D)

            Q = x @ W_q
            K = x @ W_k
            V = x @ W_v

            Q_heads = split_heads(Q)
            K_heads = split_heads(K)
            V_heads = split_heads(V)

            scores = Q_heads @ K_heads.transpose(0, 1, 3, 2) / np.sqrt(head_dim)
            weights = softmax(scores)
            x_attn = merge_heads(weights @ V_heads)
            x_out = layer_norm(x + x_attn)
            pooled = x_out.mean(axis=1)

            loss = 0
            dW_proj = np.zeros_like(W_proj)
            dW_embed = np.zeros_like(W_embed)
            dW_q = np.zeros_like(W_q)
            dW_k = np.zeros_like(W_k)
            dW_v = np.zeros_like(W_v)

            for target_i in encode(target):
                logits = pooled @ W_proj
                probs = softmax(logits)
                loss -= np.log(probs[0, target_i] + 1e-9)

                dlogits = probs
                dlogits[0, target_i] -= 1

                dW_proj += np.outer(pooled[0], dlogits[0])
                dpooled = dlogits[0] @ W_proj.T

                dx_out = np.ones_like(x_out) / T * dpooled
                d_attn = split_heads(dx_out)

                dV_heads = weights.transpose(0, 1, 3, 2) @ d_attn
                dQ_heads = d_attn @ K_heads / np.sqrt(head_dim)
                dK_heads = d_attn.transpose(0, 1, 3, 2) @ Q_heads / np.sqrt(head_dim)

                dV = merge_heads(dV_heads)
                dQ = merge_heads(dQ_heads)
                dK = merge_heads(dK_heads)

                dW_q += x.reshape(T, D).T @ dQ.reshape(T, D)
                dW_k += x.reshape(T, D).T @ dK.reshape(T, D)
                dW_v += x.reshape(T, D).T @ dV.reshape(T, D)

                dembed = dQ.reshape(T, D) @ W_q.T + dK.reshape(T, D) @ W_k.T + dV.reshape(T, D) @ W_v.T
                for i, idx in enumerate(idxs):
                    dW_embed[idx] += dembed[i]

            total_loss += loss / len(target)

            W_proj -= lr * dW_proj
            W_q -= lr * dW_q
            W_k -= lr * dW_k
            W_v -= lr * dW_v
            W_embed -= lr * dW_embed

        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss = {total_loss:.4f}")

    return W_embed, W_proj, W_q, W_k, W_v
