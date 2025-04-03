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
block_size = 8
batch_size = 32
eval_iters = 200
max_iters = 5000

# device = "mps" if torch.backends.mps.is_available() else "cpu"
device = "cpu"

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

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)

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
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=-1)
        return idx


# Training setup
torch.manual_seed(1337)
model = BigramLanguageModel(vocab_size)
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Training loop

start_time = time.time()
for steps in range(max_iters):
    # periodically evaluate the loss
    if steps % eval_iters == 0:
        losses = estimate_loss()
        print(f"step {steps}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

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
