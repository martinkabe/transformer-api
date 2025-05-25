
import torch
import json
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from fake_data_utils import generate_fake_data, FAKE_VALUE_FUNCTIONS
from bce_schema_predictor import SchemaPredictor, predict_columns


# Load model files
with open("training_data.json") as f:
    training_data = json.load(f)

# Build vocab and label map from training data
all_tokens = set()
all_columns = set()
for prompt, columns in training_data:
    all_tokens.update(["[CLS]"] + prompt)
    all_tokens.update(columns)
    all_columns.update(columns)

vocab = sorted(all_tokens)
columns_vocab = sorted(all_columns)
token_to_idx = {token: idx for idx, token in enumerate(vocab)}
column_to_idx = {col: i for i, col in enumerate(columns_vocab)}
idx_to_column = {i: col for col, i in column_to_idx.items()}
vocab_size = len(vocab)
num_columns = len(columns_vocab)


# Reload model
model = SchemaPredictor(vocab_size, num_columns)
model.load_state_dict(torch.load("schema_model.pt"))
model.eval()

# Define request schema
class PromptRequest(BaseModel):
    prompt: str
    rows: Optional[int] = 5


# FastAPI app
app = FastAPI()


@app.post("/generate")
async def generate(request: PromptRequest):
    tokens = request.prompt.lower().split()
    columns = predict_columns(model, tokens, threshold=0.4)
    columns = [col for col in columns if col in FAKE_VALUE_FUNCTIONS and col not in tokens]
    columns = list(dict.fromkeys(columns))
    return generate_fake_data(columns, request.rows)


# uvicorn fake_data_api:app --reload