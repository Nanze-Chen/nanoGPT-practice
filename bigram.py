import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameter
batch_size = 32
block_size = 8
max_iters = 5000
eval_interval = 300
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
torch.manual_seed(1337)
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from char to int
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
# encode function: get a string, turn them to a list of indexes
encode = lambda chars: [stoi[ch] for ch in chars]
# decode function: get a list of indexes, turn them to a string
decode = lambda idxes: ''.join([itos[idx] for idx in idxes])

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(len(data)*0.9)      #first 90% will be training set, the rest will be validation set
train_data = data[:n]
val_data = data[n:]

#data loading
def get_batch(split):
    data = train_data if split=='train' else val_data
    ix = torch.randint(len(data)-block_size-1, (batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, target=Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    '''one head of self-attention'''
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size)))

    def forward(self, x):
        B,T,C = x.shape     
        # In Andrej Karpathy's code, he wrote the sizes of k and q are (B,T,C), and he use (C ** -0.5)
        # to normalize wei. I think this could be problematic because the sizes of k and q are actually
        # (B,T,head_size), and head_size is a passed-in parameter. Andrej Karpathy's way of writing this
        # code will only be okay when head_size == C. Although this difference won't cause any bug, but 
        # it is not aligned with the formula, so I will change it for now.
        k = self.key(x)     #(B,T,head_size)
        q = self.query(x)   #(B,T,head_size)
        wei = q @ k.transpose(-2, -1) * (self.head_size ** -0.5) #(B,T,head_size) @ (B,head_size,T) --> ((B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T]==0, float('-inf'))  #change the upper triangle to -inf
        wei = F.softmax(wei, dim=-1) 
        # perform weighted aggregation of the value
        v = self.value(x)
        output = wei @ v
        return output   # (B,T,head_size)
    
class MultiHeadAttention(nn.Module):
    def __init__(self, head_num, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(head_num)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_size, out_size*4),
            nn.ReLU(),
            nn.Linear(out_size*4, out_size),
        )
    
    def forward(self,x):
        return self.net(x)
    
class Block(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        self.head_size = n_embd // n_heads
        self.sa_head = MultiHeadAttention(n_heads, self.head_size)
        self.ffwd = FeedForward(n_embd, n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x + self.sa_head(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# super simple bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            Block(n_embd, 4),
            Block(n_embd, 4),
            Block(n_embd, 4),
        )
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, target=None):
        B, T = idx.shape
        # idx and target are both (B, T) tensor of integers
        token_embd = self.token_embedding_table(idx) #the new shape will be (B, T, n_embd)
        pos_embd = self.position_embedding_table(torch.arange(T, device=device))   # (T, n_embd)
        x = token_embd + pos_embd           #(B,T,n_embd)
        x = self.blocks(x)                 #(B,T,C)
        logits = self.lm_head(x)            #(B,T,vocab_size)

        if target == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            target = target.view(B*T)
            loss = F.cross_entropy(logits, target)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block-size tokens
            idx_crop = idx[:, -block_size:]
            # get the prediction
            logits, loss = self(idx_crop)    #(B,T,C)
            # we only need to focus on the char at the last t since we are bigram
            focus = logits[:,-1,:]      #(B,C)
            # transform the prediction into probability distribution
            prob_distribution = F.softmax(focus, dim=-1)    #(B,C)
            # make the prediction
            new_idx = torch.multinomial(prob_distribution, num_samples=1) #(B,1)
            #concatenate the new_idx to the end of idx
            idx = torch.cat([idx, new_idx], dim=1) #(B, T+1)
        return idx
    

model = BigramLanguageModel()
model.to(device)

#create a PyTorch optimizor
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

#training loop
for iter in range(max_iters):
    #every once in a while evaluate the loss and print the result
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    #evaluate the loss
    logits, loss = model(xb,target=yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

#generate Shakespeare
context = torch.zeros((1,1), dtype=torch.long, device=device)
print('generating start----------------------------------')
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
