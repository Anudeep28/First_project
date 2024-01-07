from torch.utils.data import Dataset
import math
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from django.conf import settings


# -----------------------------------------------------------------------------

@dataclass
class ModelConfig:
    block_size: int = None  # length of the input sequences of integers
    vocab_size: int = None  # the input integers are in range [0 .. vocab_size -1]
    # parameters below control the sizes of each model slightly differently
    n_layer: int = 4
    n_embd: int = 64
    n_embd2: int = 64
    n_head: int = 4


# -----------------------------------------------------------------------------
# Transformer Language Model (*exactly* as used in GPT-2)

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)
        return y


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc=nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj=nn.Linear(4 * config.n_embd, config.n_embd),
            act=NewGELU(),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.c_proj(m.act(m.c_fc(x)))  # MLP forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x


class Transformer(nn.Module):
    """ Transformer Language Model, exactly as seen in GPT-2 """

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        # print("number of parameters: %.2fM" % (n_params/1e6,))

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (1, t, n_embd)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss


# -----------------------------------------------------------------------------
# helper functions for creating the training and test Datasets that emit words

class CharDataset(Dataset):

    def __init__(self, words, chars, max_word_length):
        self.words = words
        self.chars = chars
        self.max_word_length = max_word_length
        self.stoi = {ch: i + 1 for i, ch in enumerate(chars)}
        self.itos = {i: s for s, i in self.stoi.items()}  # inverse mapping

    def __len__(self):
        return len(self.words)

    def contains(self, word):
        return word in self.words

    def get_vocab_size(self):
        return len(self.chars) + 1  # all the possible characters and special 0 token

    def get_output_length(self):
        return self.max_word_length + 1  # <START> token followed by words

    def encode(self, word):
        ix = torch.tensor([self.stoi[w] for w in word], dtype=torch.long)
        return ix

    def decode(self, ix):
        word = ''.join(self.itos[i] for i in ix)
        return word

    def __getitem__(self, idx):
        word = self.words[idx]
        ix = self.encode(word)
        x = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        y = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        x[1:1 + len(ix)] = ix
        y[:len(ix)] = ix
        y[len(ix) + 1:] = -1  # index -1 will mask the loss at the inactive locations
        return x, y


def create_datasets(input_file):
    # preprocessing of the input text file
    with open(input_file, 'r') as f:
        data = f.read()
    words = data.splitlines()
    words = [w.strip() for w in words]  # get rid of any leading or trailing white space
    words = [w for w in words if w]  # get rid of any empty strings
    chars = sorted(list(set(''.join(words))))  # all the possible characters
    max_word_length = max(len(w) for w in words)
    # print(f"number of examples in the dataset: {len(words)}")
    # print(f"max word length: {max_word_length}")
    # print(f"number of unique characters in the vocabulary: {len(chars)}")
    # print("vocabulary:")
    # print(''.join(chars))

    # partition the input data into a training and the test set
    test_set_size = min(1000, int(len(words) * 0.1))  # 10% of the training set, or up to 1000 examples
    rp = torch.randperm(len(words)).tolist()
    train_words = [words[i] for i in rp[:-test_set_size]]
    test_words = [words[i] for i in rp[-test_set_size:]]
    # print(f"split up the dataset into {len(train_words)} training examples and {len(test_words)} test examples")

    # wrap in dataset objects
    train_dataset = CharDataset(train_words, chars, max_word_length)
    test_dataset = CharDataset(test_words, chars, max_word_length)

    return train_dataset, test_dataset


# -----------------------------------------------------------------------------
# helper functions for evaluating and sampling from the model

@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    """
    block_size = model.get_block_size()
    for _ in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        # forward the model to get the logits for the index in the sequence
        logits, _ = model(idx_cond)
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('Inf')
        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)
        # either sample from the distribution or take the most likely element
        if do_sample:
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            _, idx_next = torch.topk(probs, k=1, dim=-1)
        # append sampled index to the running sequence and continue
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


class nameGen:
    def __init__(self):
        """ samples from the model and pretty prints the decoded samples """
        self.device = 'cpu'
        self.input_file = os.path.join(settings.BASE_DIR, 'static/trial1/names.txt')
        self.model_file = os.path.join(settings.BASE_DIR, 'static/trial1/model.pt')
        # self.input_file = '/resources/names.txt'
        n_embd2, n_embd = 64, 64
        n_layer, n_head = 4, 4

        self.train_dataset, self.test_dataset = self.create_datasets()
        vocab_size = self.train_dataset.get_vocab_size()
        block_size = self.train_dataset.get_output_length()

        # load the saved model
        # init model
        config = ModelConfig(vocab_size=vocab_size, block_size=block_size,
                             n_layer=n_layer, n_head=n_head,
                             n_embd=n_embd, n_embd2=n_embd2)

        self.model = Transformer(config)
        self.model.load_state_dict(torch.load(self.model_file,map_location=torch.device(self.device)))

        # -1 because we already start with <START> token (index 0)

    def create_datasets(self):

        # preprocessing of the input text file
        with open(self.input_file, encoding="utf8") as f:
            data = f.read()
        words = data.splitlines()
        words = [w.strip() for w in words]  # get rid of any leading or trailing white space
        words = [w for w in words if w]  # get rid of any empty strings
        chars = sorted(list(set(''.join(words))))  # all the possible characters
        max_word_length = max(len(w) for w in words)
        # print(f"number of examples in the dataset: {len(words)}")
        # print(f"max word length: {max_word_length}")
        # print(f"number of unique characters in the vocabulary: {len(chars)}")
        # print("vocabulary:")
        # print(''.join(chars))

        # partition the input data into a training and the test set
        test_set_size = min(1000, int(len(words) * 0.1))  # 10% of the training set, or up to 1000 examples
        rp = torch.randperm(len(words)).tolist()
        train_words = [words[i] for i in rp[:-test_set_size]]
        test_words = [words[i] for i in rp[-test_set_size:]]
        # print(f"split up the dataset into {len(train_words)} training examples and {len(test_words)} test examples")

        # wrap in dataset objects
        train_dataset = CharDataset(train_words, chars, max_word_length)
        test_dataset = CharDataset(test_words, chars, max_word_length)

        return train_dataset, test_dataset

    def gen_name(self, num):
        top_k = -1
        top_k = top_k if top_k != -1 else None
        steps = self.train_dataset.get_output_length() - 1
        X_init = torch.zeros(num, 1, dtype=torch.long).to(self.device)
        X_samp = generate(self.model, X_init, steps, top_k=top_k, do_sample=True).to(self.device)
        train_samples, test_samples, new_samples = [], [], []
        for i in range(X_samp.size(0)):
            # get the i'th row of sampled integers, as python list
            row = X_samp[i, 1:].tolist()  # note: we need to crop out the first <START> token
            # token 0 is the <STOP> token, so we crop the output sequence at that point
            crop_index = row.index(0) if 0 in row else len(row)
            row = row[:crop_index]
            word_samp = self.train_dataset.decode(row)
            # separately track samples that we have and have not seen before
            if self.train_dataset.contains(word_samp):
                train_samples.append(word_samp)
            elif self.test_dataset.contains(word_samp):
                test_samples.append(word_samp)
            else:
                new_samples.append(word_samp)
        return new_samples
