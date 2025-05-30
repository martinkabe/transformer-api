# Title: Building a Prompt-Based Fake Dataset Generator using a Transformer from Scratch

# 1. Introduction & Goal

The goal of this project is to build a Python-based system that generates fake tabular datasets based on a text prompt like:

`"generate a school dataset"`

The system will use a transformer-like neural network (built using PyTorch) to predict a schema (i.e., column names), and then fill the table with fake data using the Faker library.

This approach simulates how large language models can be adapted for practical, structured data generation using domain-specific prompt interpretation.

A transformer is a specific kind of neural network, which lies at the core of many modern AI applications like ChatGPT, DALL-E, and translation systems. It was introduced in 2017 and excels at modeling relationships in sequential data.

Transformer architecture was introduced by a team of researchers at Google’s Brain division in a 2017 paper titled “Attention Is All You Need.” This advent of the transformer marked a seismic shift in the landscape of NLP. It ushered in the era of large language models (LLMs) that have since demonstrated remarkable advancements in the realm of NLP, surpassing the capabilities of earlier-generation recurrent neural networks (RNNs). From machine translation and sentiment analysis to question answering and text summarization, LLMs based on transformers have set new benchmarks and opened up exciting possibilities for the future of AI-driven language tasks.

**A transformer accomplishes the following:**

1. Pays Attention: Just like you might pay extra attention to important parts of a story, the transformer pays attention to important words in a sentence.

2. Understands Context: The transformer looks at all the words in the sentence together, not one after the other. This helps it understand how words depend on each other.

3. Weighs Relationships: It figures out how words are related to each other. For example, if the sentence is about a cat and a mouse, it knows that these words are connected.

4. Combines Insights: The transformer combines all this knowledge to understand the whole story and how words fit together.

5. Predicts Next Steps: With its understanding, it can even guess what words might come next in the story.


## What We Built

* A minimal transformer-based classifier that maps prompts to schemas

* A training loop using backpropagation to tune model parameters

* A fake data generator powered by the schema output

The underlying structure of your project mimics what large language models do: tokenize inputs, use embeddings and attention to learn patterns, and generate meaningful structured or unstructured output. Understanding this gives you a clear view into the architecture behind tools like ChatGPT.

# 2. Dataset & Vocabulary Preparation

## 2.1 Prompt-Response Pairs

We create a training dataset with pairs like:

```python
[
  ["generate a bank dataset", ["account_number", "name", "balance", "currency"]],
  ["create student data", ["student_id", "name", "grade", "email"]]
]
```

These examples simulate user requests and expected structured outputs.

## 2.2 Tokenization & Vocabulary

Tokenize prompts, add a `[CLS]` token, and build vocabularies for tokens and columns.

* Prompts are split into tokens (words or phrases).

* A `[CLS]` token is prepended to provide a unified representation.

* A vocabulary (vocab) is created for all tokens.

* A separate columns_vocab is built from the set of all output columns.

Tokens are embedded into vectors, creating coordinates in a high-dimensional space where semantically similar words are closer together. This embedding is the first major step in processing input in transformer models.


## 2.3 Encoding

* Prompts are encoded as sequences of integers [Prompts → sequences of integers (token indices)]

* Output columns are encoded as a multi-hot vector (multi-label classification) [Columns → multi-hot vector indicating all columns relevant to a prompt]

# 3. Model Architecture

We build a simple transformer-based classifier, named `SchemaPredictor`, which includes:

## 3.1 Embedding Layer

Each token is mapped to a vector of fixed dimension.

```python
self.embedding = nn.Embedding(vocab_size, embed_dim)
```

## 3.2 Multi-Head Attention

This allows the model to compare all tokens against each other. We use PyTorch's built-in attention to simulate a transformer block.

```python
self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
```

Attention enables the model to assign relevance between tokens. For example, the word "`model`" in different contexts (fashion vs. machine learning) can be disambiguated by attending to surrounding tokens.


## 3.3 Residual + LayerNorm

The attention output is added back to the input and normalized. Output of attention is normalized and added back to the input.

```python
self.ln = nn.LayerNorm(embed_dim)
```

## 3.4 Output Projection

Only the `[CLS]` token output is used to predict the schema.

```python
self.output_proj = nn.Linear(embed_dim, num_columns)
```

## 3.5 Output

The model returns logits for each possible column. These are passed through a sigmoid function to get probabilities.

# 4. Training

Transformers are trained by minimizing loss functions that measure the difference between predicted and actual outputs. Using binary cross-entropy allows the model to handle multiple correct labels, which is essential for predicting multiple columns in a dataset.

## 4.1 Loss Function

We use `BCEWithLogitsLoss`, a multi-label version of binary cross-entropy.

## 4.2 Optimizer

`Adam` is used for faster convergence.

## 4.3 Epoch Loop

* Each prompt is passed through the model

* Predicted logits are compared with ground truth (multi-hot vector)

* Backpropagation updates the weights

Every 50 epochs, we print the training loss.

*For each epoch:*

* Forward pass

* Compute loss

* Backpropagate errors

* Update model weights


This aligns with backpropagation. The weights (learned parameters) get updated by minimizing the loss over examples, just as in all deep learning models including GPT.


# 5. Inference

Use the trained model to predict schema columns based on new prompts and generate corresponding fake data.

Given a prompt like:

```python
"generate a hospital dataset"
```

It is encoded and passed through the model to generate probabilities for each column. Thresholding is applied to pick the most relevant columns.


## 5.1 Predict Columns

* A prompt like "generate a hospital dataset" is tokenized and encoded

* The model returns predicted probabilities for each column

* We apply a threshold (e.g., 0.4) to select columns with high confidence

## 5.2 Generate Fake Data

We use `Faker` and a predefined dictionary of functions to generate fake values for each column

```python
"name": lambda: fake.name(),
"email": lambda: fake.email(),
```

This step simulates data entries based on predicted schema.

Once a model can predict the next likely token or label, it can be used generatively by sampling from predictions. This is how models like GPT produce fluent text — token by token.


# 6. FastAPI Integration (Optional)

Deploy the model using FastAPI to provide an interface for generating fake datasets based on user prompts.

* We wrap the model in a FastAPI server

* A POST endpoint /generate takes a prompt and number of rows

* It returns a JSON response with fake data matching the predicted schema


# 7. Summary: What We Built

* A tiny transformer that maps prompts to table schemas

* A training loop that teaches the model what different domains look like

* A fake data engine that fills those tables with plausible values

This project combines natural language understanding and structured data generation in a compact, educational example of how LLM concepts can power intelligent dataset creation from scratch.


# Key Words

* Self-attention

* Embedding layers

* Multi-label classification

* Transformers in NLP

* Prompt engineering


## Key Differences Between MCP and APIs

MCP and APIs serve similar purposes — connecting systems — but they differ in ways that matter for AI.

1. Purpose
APIs are general-purpose. They were designed for software-to-software communication, not specifically for AI. MCP, on the other hand, was built from the ground up for large language models. It standardizes how LLMs fetch context and use tools.

2. Discovery
With a traditional API, you need documentation. The API doesn’t tell you what it can do; you have to already know. MCP flips this. An MCP server can be queried — “What tools do you offer?” — and it responds with a machine-readable list of functions, their inputs, and their outputs. An AI agent can adapt to new capabilities without pre-programming.

3. Standardization
Every REST API is different. One might use OAuth for authentication, another API keys, another some custom scheme. One returns XML, another JSON. MCP eliminates this variability. All MCP servers speak the same protocol, so once an AI knows how to use one, it knows how to use any of them.

4. Adaptability
If a REST API changes — a new endpoint, a modified response format — the client breaks until it’s updated. MCP avoids this because the client discovers capabilities at runtime. If a server adds a new tool, the AI can start using it immediately.


# Important Resources

## RMD Code Blocks

https://docs.readme.com/rdmd/docs/code-blocks

**Transcript:** NoteGPT_Transformers (how LLMs work) explained visually _ DL5.txt

## Create Python Virtual Environment

```bash create .venv
python3 -m venv .venv
```
```bash source .venv
source .venv/bin/activate
```

## Install Python Packages

```bash
pip3 install -r requirements.txt
```

## Run API via uvicorn

```bash
uvicorn fake_data_api:app --reload
```