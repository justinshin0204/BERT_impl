# BERT_impl


This repository is an attempt to implement **BERT** from scratch. The goal is to understand the inner workings of BERT by building it step-by-step, rather than relying on pre-built libraries. Through this process, we aim to gain a deeper understanding of its architecture and functionality.

# BERT
Before we start implementing, let's first discuss what BERT is.

BERT is a transformer-based model designed to pre-train deep bidirectional representations by considering the left and right context simultaneously in all layers using the transformer encoder architecture. 
Since its introduction in 2019, the BERT model has become the foundation for many subsequent models in the field of NLP.

The architecture is pretty similar to the original Transformer's encoder but it's slightly different.
Before we discuss the details, it's important to first understand how to train BERT, as knowing the training process is essential to understanding the modifications made to the architecture.
## Input/Output Representations
BERT receives two sentences merged into a single input sequence.
there is always a [CLS] token that represents the whole sentence. And each sentence is separated by a [SEP] token.
its form would be like " [CLS] + sentence1 +[SEP] + sentence2 + [SEP]"
![image](https://github.com/justinshin0204/BERT/assets/93083019/458a80ef-34d7-469a-9023-026586f90507)
We add **segment embeddings** to differentiate the sentences, and  **positional embeddings** to indicate the position of each token.

Concatenating makes more sense, but it's more resource-intensive.
## Pre-training
Now let's jump into two pre-training techniques
### Masked Language Modeling
We can intuitively think that bi-directional attention model would be much stronger , But since bi-directional conditioning would allow each word to indirectly “see itself” ( Because it provides more context ), We need an alternative approach.

So, the authors of the paper devised MLM.
![image](https://github.com/justinshin0204/BERT/assets/93083019/41bdfee9-fa85-4402-bbb9-88a6fc0970e3)
While using the MLM, we randomly choose tokens with a 15% probability. Among these, 80% are replaced with a mask token, 10% are replaced with a random token, and the remaining 10% are left unchanged.
( Since we don't use [MASK] in fine-tuning, we have to put some original sentence itself )

### NSP

Next Sentence Prediction (NSP) is another crucial task used during BERT's pre-training to understand the relationship between two sentences.
![image](https://github.com/justinshin0204/BERT/assets/93083019/7309bf78-94ba-44f9-9d78-385c11cab5ba)
The fully connected layer is attached to the hidden state of the [CLS] token and predicts if the second sentence is connected to the first one or not.


## Model architecture
![image](https://github.com/justinshin0204/BERT/assets/93083019/46c633a7-9690-4714-8899-33ab883e267c)
Now, let's discuss the model architecture: it consists of 12 layers, 12 attention heads, and a word dimension of 768.
Each block is the same as a Transformer encoder, but the final part differs due to the MLM and NSP tasks.

We've briefly discussed about BERT.
Now, lets move on to the implementation part



# Implementation


## Installation

To install the required datasets package, run the following command:

```bash
pip install transformers torch datasets einops
```
Import the necessary modules in your python script
```py
from transformers import BertTokenizer
import pandas as pd
from torch import nn, optim
from datasets import load_dataset
import math, random
from torch.nn.utils.rnn import pad_sequence
import time
import torch
from einops import rearrange
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
```

## Load BERT Tokenizer

To load the BERT tokenizer and obtain the special token IDs, you can use the following code:

```python
# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Get special token IDs
pad_idx = tokenizer.pad_token_id
mask_idx = tokenizer.mask_token_id
sep_idx = tokenizer.sep_token_id
cls_idx = tokenizer.cls_token_id

print("pad_idx =", pad_idx)
print("mask_idx =", mask_idx)
print("sep_idx =", sep_idx)
print("cls_idx =", cls_idx)
```

## Setting the hyperparameters

```py
vocab_size = tokenizer.vocab_size
BATCH_SIZE = 256
LAMBDA = 0.01 # l2 Regularization
EPOCH = 40
max_len = 512
criterion = nn.CrossEntropyLoss(ignore_index = -100) # ignore the unmasked tokens

n_layers = 12
d_model = 768
d_ff = 3012
n_heads = 12
drop_p = 0.1

### Linear Scheduler ###
warmup_steps = 10000
LR_peak = 1e-4

save_model_path = "you should fill this part on your own"
save_history_path = "you should fill this part on your own"
```
I set up the hyperparameters for BERT base because BERT large was too heavy, but even BERT base is still too heavy.😅
you can change the setups if you want
## Dataset preprocessing
```py
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
def split_sentences(text):
    # separate the sentence
    sentences = text.split(". ")
    return sentences

# extract the 'text' part and put it in the list
texts = []
for example in dataset['train']:
    sentences = split_sentences(example['text'])
    texts.extend(sentences)

texts=texts[:10000]
data=texts
```
We split the sentence by the dot ('.')  and put it in to a list named texts.
This is a 1D array, which makes handling the data easier.


## Define the DataLoader
```py
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def mask_tokens(self, sentence):
        #Masking probability
        MASK_PROB = 0.15
        # [CLS]+sentence1 +[SEP]+ sentence2 +[SEP] ==> tokenize
        input_ids = tokenizer.encode(sentence, truncation=True, max_length=max_len, add_special_tokens=False)

        segment_ids = []
        labels = []
        is_second_sentence = False
        for i, token in enumerate(input_ids):
            if token == sep_idx:
                is_second_sentence = True
            segment_ids.append(0 if not is_second_sentence else 1)
            # Random masking
            val = random.random()
            if val <= MASK_PROB and token not in {tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id}:
                labels.append(token)
                val = random.random()
                if val < 0.8: # with 80% probability, replace the token with a [MASK] token
                    input_ids[i] = mask_idx
                elif 0.8 <= val < 0.9: # with 10% probability, replace the token with a random token
                    input_ids[i] = random.choice(list(tokenizer.get_vocab().values()))
            else:
                labels.append(-100)

        return torch.tensor(input_ids), torch.tensor(labels), torch.tensor(segment_ids)

    def __getitem__(self, idx):
        sentence1 = self.data[idx]

        if random.random() > 0.5:  # 50% chance to use consecutive sentences
            next_idx = idx + 1 if idx + 1 < len(self.data) else 0
            sentence2 = self.data[next_idx]
            nsp_label = torch.tensor(1)  # True next sentence
        else:  # 50% chance to use random sentence
            random_idx = random.randint(0, len(self.data) - 1)
            sentence2 = self.data[random_idx]
            nsp_label = torch.tensor(0)  # Not a true next sentence

        combined_sentence = '[CLS]' + sentence1 + '[SEP]' + sentence2 + '[SEP]'

        input_ids, mtp_label, segment_ids = self.mask_tokens(combined_sentence)

        return input_ids, mtp_label, nsp_label, segment_ids
```
Through the **`getitem`** function, we combine two sentences. If a generated random number is greater than 0.5, we append a random sentence and set the NSP label to 0. If the number is less than or equal to 0.5, we append a random sentence and set the NSP label to 1.

The **`mask_tokens`** function takes the combined form of the two sentences, performs masking, and outputs the segment embeddings, token embeddings, and MTP labels.

When an article changes, the following sentence might not be connected. However, due to the length of the paragraph, we decided to disregard this effect
```py
def custom_collate_fn(batch):
    input_ids = [item[0] for item in batch]
    mtp_labels = [item[1] for item in batch]
    nsp_labels = [item[2] for item in batch]
    segment_ids = [item[3] for item in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    mtp_labels = pad_sequence(mtp_labels, batch_first=True, padding_value=-100)
    segment_ids = pad_sequence(segment_ids, batch_first=True, padding_value=0)

    nsp_labels = torch.stack(nsp_labels)

    return input_ids, mtp_labels, nsp_labels, segment_ids

# Assuming your data is a list of sentences
data =texts  #[sentence1 , sentence 2, sentence 3 ...] 

custom_DS = CustomDataset(data)

train_DS, val_DS, test_DS= torch.utils.data.random_split(custom_DS, [9700, 200, 100])

train_DL = torch.utils.data.DataLoader(train_DS, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
val_DL = torch.utils.data.DataLoader(val_DS, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
test_DL = torch.utils.data.DataLoader(test_DS, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)

print(len(train_DS))
print(len(val_DS))
print(len(test_DS))
```
I couldn't find it in the paper, but according to various blogs, the padding for segment_id is usually set to 0. With this padding, we generate the DL as described above
# Model architecture

## Multi head attention 
```py
class MHA(nn.Module):
    def __init__(self, d_model, n_heads, drop_p):
        super().__init__()

        self.n_heads = n_heads

        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(drop_p)
        self.scale = torch.sqrt(torch.tensor(d_model / n_heads))

    def forward(self, x, mask=None):
        Q = self.fc_q(x)  # numworddim
        K = self.fc_k(x)
        V = self.fc_v(x)

        Q = rearrange(Q, 'num word (head dim) -> num head word dim', head=self.n_heads)
        K = rearrange(K, 'num word (head dim) -> num head word dim', head=self.n_heads)
        V = rearrange(V, 'num word (head dim) -> num head word dim', head=self.n_heads)

        attention_score = Q @ K.transpose(-2, -1) / self.scale 

        if mask is not None:
            attention_score[mask] = -1e10
        attention_weights = torch.softmax(attention_score, dim=-1) 

        attention_weights = self.dropout(attention_weights)

        attention = attention_weights @ V

        x = rearrange(attention, 'num head word dim -> num word (head dim)')
        x = self.fc_o(x)  # numworddim

        return x, attention_weights
```
I think the `rearrange` function is absolutely beautiful
I'm sure the image below will help you understand the code clearly ☺️.



![image](https://github.com/justinshin0204/BERT/assets/93083019/84474740-3002-4b95-9a41-3fd64287db55)

## Feedforward
```py
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, drop_p):
        super().__init__()

        self.linear = nn.Sequential(nn.Linear(d_model, d_ff),
                                    nn.GELU(),
                                    nn.Dropout(drop_p),
                                    nn.Linear(d_ff, d_model))

    def forward(self, x):
        x = self.linear(x)
        return x
```
This part is quite straightforward.
Remembert that we use GELU as an activation function in BERT.

## Encoderlayer
```py
class EncoderLayer(nn.Module): # attention -> drop -> add -> norm
    def __init__(self, d_model, d_ff, n_heads, drop_p):
        super().__init__()

        self.self_atten = MHA(d_model, n_heads, drop_p)
        self.self_atten_LN = nn.LayerNorm(d_model)

        self.FF = FeedForward(d_model, d_ff, drop_p)
        self.FF_LN = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(drop_p)

    def forward(self, x, enc_mask):
        residual = x

        atten_output, _ = self.self_atten(x, enc_mask)
        atten_output = self.dropout(atten_output)
        x = residual + atten_output
        x = self.self_atten_LN(x)

        residual = x
        ff_output = self.FF(x)
        ff_output = self.dropout(ff_output)
        x = residual + ff_output
        x = self.FF_LN(x)

        return x
```
This EncoderLayer class implements a Transformer encoder layer where each sub-layer follows the order: attention -> dropout -> add -> layer normalization.

## Encoder
```py
class Encoder(nn.Module):
    def __init__(self, vocab_size, max_len, n_layers, d_model, d_ff, n_heads, drop_p):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.seg_embedding = nn.Embedding(2, d_model)

        self.dropout = nn.Dropout(drop_p)

        self.layers = nn.ModuleList([EncoderLayer(d_model, d_ff, n_heads, drop_p) for _ in range(n_layers)])

        self.LN_out = nn.LayerNorm(d_model)

    def forward(self, x, seg, enc_mask, atten_map_save = False):

        pos = torch.arange(x.shape[1]).expand_as(x).to(DEVICE) 

        x = self.token_embedding(x) + self.pos_embedding(pos) + self.seg_embedding(seg) 
        x = self.dropout(x)

        for layer in self.layers:
            x= layer(x, enc_mask)

        return x
```
we sum all the embeddings
## BERT
```py
class BERT(nn.Module):
    def __init__(self, vocab_size, max_len, n_layers, d_model, d_ff, n_heads, drop_p):
        super().__init__()

        self.encoder = Encoder(vocab_size, max_len, n_layers, d_model, d_ff, n_heads, drop_p)

        self.n_heads = n_heads

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)

    def make_enc_mask(self, x): 

        enc_mask = (x == pad_idx).unsqueeze(1).unsqueeze(2) 
        enc_mask = enc_mask.expand(x.shape[0], self.n_heads, x.shape[1], x.shape[1])
        """ pad mask 
        F F T T
        F F T T
        F F T T
        F F T T
        """
        return enc_mask

    def forward(self, x, seg, atten_map_save = False):

        enc_mask = self.make_enc_mask(x)

        out= self.encoder(x, seg, enc_mask, atten_map_save = atten_map_save)

        return out


## BERT_LM
class BERT_LM(nn.Module): 
    def __init__(self, bert, vocab_size, d_model):
        super().__init__()

        self.bert = bert

        self.nsp = nn.Linear(d_model, 2) # NSP: Next Sentence Prediction
        self.mtp = nn.Linear(d_model, vocab_size) # MTP: Masked Token Prediction

        nn.init.normal_(self.nsp.weight, mean=0, std=0.02)
        nn.init.normal_(self.mtp.weight, mean=0, std=0.02)

    def forward(self, x, seg, atten_map_save = False):

        x = self.bert(x, seg, atten_map_save)

        return self.nsp(x[:,0]), self.mtp(x)
```
`make_enc_mask` masks the pad_sequence.
# Training
Now its time to finally test your model!

```py
def Train(model, train_DL, val_DL, criterion, optimizer, scheduler = None):
    loss_history = {"train": [], "val": []}
    best_loss = 9999
    for ep in range(EPOCH):
        epoch_start = time.time()

        model.train() # train mode
        train_loss = loss_epoch(model, train_DL, criterion, optimizer = optimizer, scheduler = scheduler)
        loss_history["train"] += [train_loss]

        model.eval() # test mode
        with torch.no_grad():
            val_loss = loss_epoch(model, val_DL, criterion)
            loss_history["val"] += [val_loss]
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({"model": model,
                            "ep": ep,
                            "optimizer": optimizer,
                            "scheduler": scheduler,}, save_model_path)
        # print loss
        print(f"Epoch {ep+1}: train loss: {train_loss:.5f}   val loss: {val_loss:.5f}   current_LR: {optimizer.param_groups[0]['lr']:.8f}   time: {time.time()-epoch_start:.0f} s")
        print("-" * 20)

    torch.save({"loss_history": loss_history,
                "EPOCH": EPOCH,
                "BATCH_SIZE": BATCH_SIZE}, save_history_path)

def Test(model, test_DL, criterion):
    model.eval() # test mode
    with torch.no_grad():
        test_loss = loss_epoch(model, test_DL, criterion)
    print(f"Test loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):.3f}")

def loss_epoch(model, DL, criterion, optimizer = None, scheduler = None):
    N = len(DL.dataset) # the number of data

    rloss=0
    for x_batch, mtp_label, nsp_label, seg in tqdm(DL, leave=False):
        x_batch = x_batch.to(DEVICE)
        mtp_label = mtp_label.to(DEVICE)
        nsp_label = nsp_label.to(DEVICE)
        seg = seg.to(DEVICE)
        # inference
        y_hat_NSP = model(x_batch, seg)[0]
        y_hat_MTP = model(x_batch, seg)[1]
        # Loss for NSP, MTP
        nsp_loss = criterion(y_hat_NSP, nsp_label)
        mtp_loss = criterion(y_hat_MTP.permute(0,2,1), mtp_label) 
        loss = nsp_loss + mtp_loss
        # update
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        # loss accumulation
        rloss += loss.item() * x_batch.shape[0]
    loss_e = rloss/N
    return loss_e
```
## Make the model
```py
from transformers import get_scheduler
params = [p for p in model.parameters() if p.requires_grad] 
optimizer = optim.Adam(nn.Linear(1, 1).parameters(), lr=0)
scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=int(len(train_DS)*EPOCH/BATCH_SIZE)
)


bert = BERT(vocab_size, max_len, n_layers, d_model, d_ff, n_heads, drop_p).to(DEVICE)
model = BERT_LM(bert, vocab_size, d_model).to(DEVICE)
Train(model, train_DL, val_DL, criterion, optimizer, scheduler)

torch.save(model.state_dict(), save_model_path)
```
