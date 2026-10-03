# PyTorch in One Hour — Study Notes

> Personal study notes based on **"PyTorch in One Hour: From Tensors to Training Neural Networks on Multiple GPUs"** by Sebastian Raschka (Jul 2025).
> Original tutorial: https://sebastianraschka.com/teaching/pytorch-1h/
> These notes summarize the concepts in my own words; read the original for full explanations, figures, and outputs.

---

## Contents

1. What PyTorch is
2. Tensors
3. Computation graphs
4. Autograd
5. Building models with `nn.Module`
6. Datasets & DataLoaders
7. The training loop
8. Saving & loading
9. GPUs: single and multi-GPU (DDP)
10. Quick reference & gotchas
11. Further reading

---

## 1. What PyTorch Is

**Three core components:**

| Component | What it gives you |
|---|---|
| **Tensor library** | NumPy-like arrays that can run on CPU *or* GPU |
| **Autograd engine** | Automatic gradient computation for backpropagation |
| **Deep learning utilities** | Layers, loss functions, optimizers, data loading |

**Terminology hierarchy:** AI ⊃ Machine Learning ⊃ Deep Learning (deep neural nets) — LLMs live inside deep learning.

**Supervised learning workflow:** train on labeled data → evaluate → predict labels for new data.

### Installation tips
- Use a Python version 1–2 releases behind the latest (better library support).
- `pip install torch` — or use the selector at pytorch.org for a specific CUDA build.
- Sanity checks:
  - `torch.__version__`
  - `torch.cuda.is_available()` → NVIDIA GPU
  - `torch.backends.mps.is_available()` → Apple Silicon
- No GPU? Google Colab offers free, time-limited GPU runtimes.

---

## 2. Tensors

- A tensor generalizes scalars/vectors/matrices to N dimensions.
  - Rank 0 = scalar, rank 1 = vector, rank 2 = matrix, rank 3+ = "3D tensor", etc.
- Think of them as **data containers** with GPU support + gradient tracking.

### Creating tensors
```python
import torch

s = torch.tensor(7)                  # 0D
v = torch.tensor([1, 2, 3])          # 1D
m = torch.tensor([[1, 2], [3, 4]])   # 2D
```

### Data types
- Python ints → `torch.int64`
- Python floats → `torch.float32` (default; good precision/speed trade-off, GPU-optimized)
- Convert with `.to()`:
```python
v_float = v.to(torch.float32)
```

### Essential operations
| Operation | Purpose |
|---|---|
| `.shape` | Dimensions |
| `.reshape(r, c)` / `.view(r, c)` | Change shape (`.view` is very common) |
| `.T` | Transpose (not the same as reshape!) |
| `a.matmul(b)` or `a @ b` | Matrix multiplication |

> Why duplicate APIs? PyTorch inherited Lua Torch conventions, then added NumPy-style aliases.

---

## 3. Models as Computation Graphs

- A **computation graph** is a directed graph of the operations that produce an output (and loss).
- PyTorch builds this graph **dynamically in the background** as you run operations.
- The graph is what makes gradient computation possible.

Example idea — logistic regression forward pass:
`input × weight + bias → sigmoid → compare with label → loss`

---

## 4. Autograd (Automatic Differentiation)

- Set `requires_grad=True` on parameters you want gradients for.
- Backpropagation = applying the **chain rule** from the loss backward through the graph.

### Two ways to get gradients
```python
import torch.nn.functional as F

w = torch.tensor([1.5], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)
x, y = torch.tensor([2.0]), torch.tensor([1.0])

loss = F.binary_cross_entropy(torch.sigmoid(x * w + b), y)

# Option A: manual (debugging / experiments)
# from torch.autograd import grad
# dw = grad(loss, w, retain_graph=True)

# Option B: the usual way
loss.backward()
print(w.grad, b.grad)
```

**Key points**
- The graph is freed after computing gradients unless `retain_graph=True`.
- `.backward()` fills `.grad` on all leaf tensors.
- Takeaway: you rarely compute derivatives by hand — PyTorch does the calculus.

---

## 5. Building Neural Networks with `nn.Module`

### Pattern
- Subclass `torch.nn.Module`
- Define layers in `__init__`
- Define data flow in `forward`
- `backward` is generated automatically

```python
class MLP(torch.nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_in, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, n_out),   # outputs raw logits
        )

    def forward(self, x):
        return self.net(x)
```

### Useful inspections
- `print(model)` — architecture summary
- Count trainable params:
  ```python
  sum(p.numel() for p in model.parameters() if p.requires_grad)
  ```
- Access a layer's weights: `model.net[0].weight`, `.bias`, `.shape`

### Concepts worth remembering
- **`nn.Linear`** = weight matrix multiply + bias ("fully connected" / "feedforward" layer).
- **Random init** breaks symmetry so neurons learn different things.
- **Reproducibility:** `torch.manual_seed(123)` before creating the model.
- **`grad_fn`** on an output (e.g. `AddmmBackward0`) shows the last op recorded in the graph.

### Inference best practices
- Wrap prediction in `torch.no_grad()` → saves memory & compute.
- Models return **logits** (no softmax) because loss functions like `cross_entropy` apply softmax internally for numerical stability.
- Want probabilities? Call `torch.softmax(logits, dim=1)` explicitly.

---

## 6. Efficient Data Loading

**Two classes:**
- **`Dataset`** — how to fetch *one* example
- **`DataLoader`** — shuffling, batching, parallel loading

### Custom Dataset: three required methods
| Method | Role |
|---|---|
| `__init__` | Store data / paths / connections |
| `__getitem__(i)` | Return one `(features, label)` pair |
| `__len__` | Number of examples |

```python
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __getitem__(self, i):
        return self.X[i], self.y[i]
    def __len__(self):
        return len(self.y)

loader = DataLoader(MyDataset(X, y), batch_size=32,
                    shuffle=True, drop_last=True, num_workers=4)
```

### DataLoader settings cheat sheet
| Setting | Guidance |
|---|---|
| `shuffle=True` | For training; order changes each epoch (avoids repetitive update cycles) |
| `shuffle=False` | For validation/test |
| `drop_last=True` | Drops a small final batch that can hurt convergence |
| `num_workers=0` | Loads in main process — can starve the GPU on big jobs |
| `num_workers>0` | Parallel loading; ~4 is often a good start. Avoid in Jupyter / tiny datasets |

**Class labels** must start at 0 and go up to `num_output_nodes - 1`.

**Epoch** = one full pass over the training set.

---

## 7. A Typical Training Loop

```python
model = MLP(n_in=2, n_out=2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

for epoch in range(num_epochs):
    model.train()
    for features, labels in train_loader:
        logits = model(features)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()   # 1. clear old gradients
        loss.backward()         # 2. compute new gradients
        optimizer.step()        # 3. update parameters

    model.eval()
    # evaluate on validation data here
```

### The core three-step rhythm
1. `optimizer.zero_grad()` — **don't forget**, or gradients accumulate
2. `loss.backward()`
3. `optimizer.step()`

### Train vs eval mode
- `model.train()` / `model.eval()` matter for **dropout** and **batch norm**.
- Include them always — protects you when the architecture changes later.

### Hyperparameters & data splits
- Learning rate and number of epochs are **hyperparameters** — tuned by watching the loss.
- **Validation set:** used many times to tune hyperparameters.
- **Test set:** use once, at the end, for an unbiased estimate.

### From logits to predictions
- `torch.argmax(logits, dim=1)` → predicted class (softmax not required for argmax).
- Accuracy = `(preds == labels).sum() / len(labels)`.
- For large data, compute accuracy **batch-by-batch** through a DataLoader:

```python
def accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total
```

---

## 8. Saving & Loading Models

**Recommended:** save the `state_dict` (parameter dictionary), not the whole object.

```python
# Save
torch.save(model.state_dict(), "model.pth")

# Load
model = MLP(2, 2)  # architecture must match exactly
model.load_state_dict(torch.load("model.pth", weights_only=True))
```

- `.pth` / `.pt` are conventional extensions.
- You need a model instance in memory before loading weights.

---

## 9. Training on GPUs

### 9.1 Device basics
- A **device** is where tensors live and ops run (`"cpu"`, `"cuda"`, `"cuda:1"`, `"mps"`).
- Move with `.to(device)`.
- **All tensors in an operation must be on the same device** — otherwise `RuntimeError`.

### 9.2 Single-GPU training — only 3 changes
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Apple Silicon: "mps" if torch.backends.mps.is_available() else "cpu"

model.to(device)                                  # change 1 & 2
for features, labels in train_loader:
    features, labels = features.to(device), labels.to(device)  # change 3
    ...
```
- Tiny models may not speed up (transfer overhead); big models / LLMs benefit hugely.

### 9.3 Multi-GPU with DistributedDataParallel (DDP)

**How DDP works**
1. One process per GPU, each with a **full model copy**.
2. `DistributedSampler` gives each GPU a **different, non-overlapping** slice of data.
3. Each GPU runs forward + backward independently.
4. Gradients are **averaged and synced** across GPUs → all copies stay identical.
5. Result: near-linear speedup (2 GPUs ≈ 2× faster per epoch, minus communication overhead).

**Key building blocks**
| Piece | Role |
|---|---|
| `init_process_group(backend=...)` | Start distributed communication (`nccl` on Linux/NVIDIA, `gloo` on Windows) |
| `torch.cuda.set_device(rank)` | Pin each process to its GPU |
| `DistributedSampler(dataset)` | Split data across processes (set `shuffle=False` on the DataLoader) |
| `DDP(model, device_ids=[rank])` | Wrap the model; original is at `model.module` |
| `sampler.set_epoch(epoch)` | Different shuffle order each epoch |
| `destroy_process_group()` | Clean shutdown |

**Skeleton**
```python
import os
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

def main(rank, world_size):
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    loader = DataLoader(ds, batch_size=32, shuffle=False,
                        sampler=DistributedSampler(ds), pin_memory=True)
    model = DDP(MLP(2, 2).to(rank), device_ids=[rank])
    # ... standard training loop, calling loader.sampler.set_epoch(epoch) ...

    destroy_process_group()

if __name__ == "__main__":
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("LOCAL_RANK", 0))
    main(rank, world_size)
```

**Launching with `torchrun`** (handles process spawning + env vars like `RANK`, `LOCAL_RANK`, `WORLD_SIZE`):
```bash
torchrun --nproc_per_node=2 train_ddp.py
torchrun --nproc_per_node=$(nvidia-smi -L | wc -l) train_ddp.py   # all GPUs
```

**DDP gotchas**
- ❌ Doesn't work in Jupyter — run as a script.
- Duplicate log lines → each process prints. Guard with `if rank == 0:`.
- Tiny datasets may leave some GPUs with zero batches → enlarge data for more GPUs.
- Model too big for one GPU? Look at **FSDP** (Fully Sharded Data Parallel).

---

## 10. Quick Reference & Gotchas

**Checklist for any training script**
- [ ] `torch.manual_seed(...)` for reproducibility
- [ ] Model returns logits; use `F.cross_entropy`
- [ ] `optimizer.zero_grad()` → `loss.backward()` → `optimizer.step()`
- [ ] `model.train()` / `model.eval()` toggled appropriately
- [ ] `torch.no_grad()` for inference/evaluation
- [ ] Model **and** data on the same device
- [ ] Save with `state_dict()`

**Common errors**
| Symptom | Likely cause |
|---|---|
| "Expected all tensors to be on the same device" | Forgot `.to(device)` on data or model |
| Loss explodes / weird training | Missing `zero_grad()` (gradient accumulation) |
| Label index errors in loss | Labels not in `0 … num_classes-1` |
| Notebook crash with DataLoader | `num_workers > 0` in Jupyter |
| Slow GPU training | Data loading bottleneck → raise `num_workers` |

> Why isn't `zero_grad()` automatic? Because sometimes you *want* to accumulate gradients (e.g., gradient accumulation to simulate larger batches on limited memory).

---

## 11. Further Reading (from the tutorial)

- *Machine Learning with PyTorch and Scikit-Learn* — Raschka, Liu, Mirjalili (2022)
- *Deep Learning with PyTorch* — Stevens, Antiga, Viehmann (2021)
- Video: Tensors in Deep Learning — https://www.youtube.com/watch?v=JXfDlgrfOBY
- Model Evaluation, Selection & Algorithm Selection — https://arxiv.org/abs/1811.12808
- Introduction to Calculus (free PDF) — https://sebastianraschka.com/pdf/supplementary/calculus.pdf
- Gradient accumulation for LLM finetuning — https://sebastianraschka.com/blog/2023/llm-grad-accumulation.html
- PyTorch FSDP intro — https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/
- DDP API docs — https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
