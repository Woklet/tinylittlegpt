config = {
    "vocab_size": 50257,       # Will match tokenizer
    "context_length": 512,     # How many tokens it can look at
    "embedding_dim": 512,
    "num_heads": 8,
    "num_layers": 6,
    "ffn_hidden_dim": 2048,    # Size of MLP hidden layer
    "dropout": 0.1,
    "block_size": 128,         # Size of input blocks for training
}
