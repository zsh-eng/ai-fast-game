import torch
import torch.nn as nn
from torch.nn import functional as F
import time

# Read the input text
with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Create the character level encoding
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Create the encoding/decoding mappings
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(string: str) -> list[int]:
    return [stoi[c] for c in string]


def decode(token_list: list[int]) -> str:
    return "".join([itos[i] for i in token_list])


# Create the training data
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(text))
train_data = data[:n]
val_data = data[n:]

# Hyperparameters
block_size = 256
batch_size = 64
# eval_iters = 200
# max_iters = 200
eval_iters = 50
max_iters = 1000
learning_rate = 3e-4
n_embed = 384

nheads = 6
nlayers = 6  # number of block sandwiches
dropout = 0.2

device = "mps" if torch.backends.mps.is_available() else "cpu"
# device = "cpu"

print(f"Using device: {device}")


# Data loading
def get_batch(split):
    data = train_data if split == "train" else val_data
    random_indices = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in random_indices])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in random_indices])

    x, y = x.to(device), y.to(device)

    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    model.train()
    return out


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, C)
        q = self.query(x)  # (B, T, C)

        # compute attention weights ("affinities")
        weights = q @ k.transpose(-2, -1) * C**-0.5
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        v = self.value(x)

        out = weights @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)

        return out


class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embed, num_heads):
        super().__init__()
        head_size = n_embed // num_heads

        self.sa = MultiHeadAttention(num_heads, head_size)
        self.ffwd = FeedForward(n_embed)

        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)

        self.blocks = nn.Sequential(
            Block(n_embed, num_heads=4),
            Block(n_embed, num_heads=4),
            Block(n_embed, num_heads=4),
            nn.LayerNorm(n_embed),
        )

        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device)
        )  # (T, C)
        x = tok_emb + pos_emb
        x = self.blocks(x)

        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # last block_size tokens
            idx_cond = idx[:, -block_size:]

            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=-1)
        return idx


# Training setup
torch.manual_seed(1337)
model = BigramLanguageModel()
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop

start_time = time.time()
for steps in range(max_iters):
    # periodically evaluate the loss
    if steps % eval_iters == 0:
        losses = estimate_loss()
        print(
            f"step {steps}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


total_time = time.time() - start_time
print(f"Training completed in {total_time:.2f} seconds")

# Generate some text
if __name__ == "__main__":
    NEWLINE_IDX = stoi["\n"]
    idx = torch.tensor([[NEWLINE_IDX]], dtype=torch.long, device=device)
    print(decode(model.generate(idx, max_new_tokens=300)[0].tolist()))
