import torch
import torch.nn as nn
from torch.nn import functional as F

#hyperparameters
batch_size = 64
block_size = 64
embedding_dim = 384
n_heads = 6
max_iters = 5000
eval_interval = 500
eval_iters = 200
learning_rate = 3e-4
dropout_rate = 2e-1
device = "cuda" if torch.cuda.is_available() else "cpu"
    

class Head(nn.Module):
    def __init__(self,n_embeddings,head_size,block_size):
        super().__init__()
        self.key = nn.Linear(n_embeddings,head_size,bias=False)
        self.query = nn.Linear(n_embeddings,head_size,bias=False)
        self.value = nn.Linear(n_embeddings,head_size,bias=False)
        self.tril = torch.tril(torch.ones(block_size,block_size))
        self.head_size = head_size
        self.drop = nn.Dropout(dropout_rate)
    
    def forward(self,x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        wei = q @ k.transpose(-2,-1) * self.head_size**-0.5
        wei = wei.masked_fill(self.tril == 0, float('-inf'))
        wei = F.softmax(wei,dim=-1)
        wei = self.drop(wei)
        out = wei @ v
        return out

class FFW(nn.Module):
    def __init__(self,n_embeddings):
        super().__init__()
        self.l1 = nn.Linear(n_embeddings,4 *n_embeddings)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(4*n_embeddings,n_embeddings)
        self.drop = nn.Dropout(dropout_rate)

    def forward(self,x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.drop(x)
        return x

class MultiHead(nn.Module):
    def __init__(self,n_heads,n_embeddings,head_size,block_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embeddings,head_size,block_size) for _ in range(n_heads)])
        self.projection = nn.Linear(n_heads*head_size,n_embeddings,bias=False)
        self.drop = nn.Dropout(dropout_rate)

    def forward(self,x):
        x = torch.cat([head(x) for head in self.heads],dim=-1)
        x = self.projection(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self,n_heads,n_embeddings,block_size):
        super().__init__()
        head_size = n_embeddings // n_heads
        self.lm_att = FFW(n_embeddings)
        self.mh_att = MultiHead(n_heads,n_embeddings,head_size,block_size)
        self.ln1 = nn.LayerNorm(n_embeddings)
        self.ln2 = nn.LayerNorm(n_embeddings)

    def forward(self,x):
        x = x + self.mh_att(self.ln1(x))
        x = x + self.lm_att(self.ln2(x))
        return x

class BigramModel(nn.Module):
    def __init__(self,vocab_size,embedding_dim,n_heads,block_size):
        super().__init__()
        self.token_emb_table = nn.Embedding(vocab_size, embedding_dim)
        self.pos_emb_table = nn.Embedding(block_size,embedding_dim)
        self.blocks = nn.Sequential(
            Block(n_heads,embedding_dim,block_size),
            Block(n_heads,embedding_dim,block_size),
            Block(n_heads,embedding_dim,block_size),
            nn.LayerNorm(embedding_dim)
        )
        self.lm_head = nn.Linear(embedding_dim,vocab_size)

    def forward(self,idxs,targets=None):
        """Input is size Batch, Time """
        B,T = idxs.shape

        token_emb = self.token_emb_table(idxs)
        pos_emb = self.pos_emb_table(torch.arange(T,device=device))
        x = token_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(x)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


def get_batch(split):
    data = train_data if split=="train" else val_data
    ix = torch.randint(0,len(data)-block_size,(batch_size,))
    xs = torch.stack([data[i:i+block_size] for i in ix])
    ys = torch.stack([data[i+1:i+block_size+1] for i in ix])
    xs, ys = xs.to(device),ys.to(device)
    return xs,ys

@torch.no_grad()
def estimate_loss(): 
    out = {}
    model.eval()
    for split in ["train","eval"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x,y = get_batch(split)
            logits,loss = model(x,y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

#Getting the data 
with open("input.txt", 'r', encoding='utf-8') as f:
    data = f.read()
n = len(data)

#Processing the data
chars = sorted(list(set(data)))
vocab_len = len(chars)
#create the "tokenizer" (we are working on character level so 1 char = 1 number"
stoi = {ch:i for i,ch in enumerate(chars)}
itoc = {i:ch for i,ch in enumerate(chars)}
encode = lambda s:[stoi[c] for c in s]
decode = lambda arr:"".join([itoc[num] for num in arr])

#Train/Eval splits
encoded_data = torch.tensor(encode(data))
train_data = encoded_data[:int(n*0.9)]
val_data = encoded_data[int(n*0.9):]

model = BigramModel(vocab_len,embedding_dim,n_heads,block_size)
m = model.to(device)

#Adam optimizer
optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate)

for iter in range(max_iters):
    if ( (iter%eval_interval == 0 or iter == max_iters-1) and iter!=0):
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']}, val loss {losses['eval']}")
    #Sample
    xb,yb = get_batch("train")
 
    #Loss
    logits, loss = m(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    print("loss",loss," epoch = ", iter)
    optimizer.step()

context = torch.zeros((1,1),dtype=torch.long,device=device)
print(decode(m.generate(context,max_new_tokens=500)[0].tolist()))
   
