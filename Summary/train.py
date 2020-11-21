import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch import cuda

device = 'cuda' if cuda.is_available() else 'cpu'


class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, source_len, summ_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = summ_len
        self.text = self.data.text
        self.ctext = self.data.ctext

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        ctext = str(self.ctext[index])
        ctext = ' '.join(ctext.split())

        text = str(self.text[index])
        text = ' '.join(text.split())

        source = self.tokenizer.batch_encode_plus([ctext], max_length=self.source_len, pad_to_max_length=True,
                                                  return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([text], max_length=self.summ_len, pad_to_max_length=True,
                                                  return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }


def train(epoch, tokenizer, model, device, loader, optimizer):
    model.train()
    for _, data in enumerate(loader, 0):
        y = data['target_ids'].to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data['source_ids'].to(device, dtype=torch.long)
        mask = data['source_mask'].to(device, dtype=torch.long)

        outputs = model(input_ids=ids, attention_mask=mask, decoder_input_ids=y_ids, lm_labels=lm_labels)
        loss = outputs[0]

        if _ % 500 == 0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # xm.optimizer_step(optimizer)
        # xm.mark_step()


TRAIN_BATCH_SIZE = 2  # input batch size for training (default: 64)
VALID_BATCH_SIZE = 2  # input batch size for testing (default: 1000)
TRAIN_EPOCHS = 2  # number of epochs to train (default: 10)
VAL_EPOCHS = 1
LEARNING_RATE = 1e-4  # learning rate (default: 0.01)
SEED = 42  # random seed (default: 42)
MAX_LEN = 512
SUMMARY_LEN = 150

# Set random seeds and deterministic pytorch for reproducibility
torch.manual_seed(SEED)  # pytorch random seed
np.random.seed(SEED)  # numpy random seed
torch.backends.cudnn.deterministic = True

tokenizer = T5Tokenizer.from_pretrained("t5-base")

df = pd.read_csv('./dataset/news_summary.csv', encoding='latin-1')
df = df[['text', 'ctext']]
df.ctext = 'summarize: ' + df.ctext

print(df.head())

train_size = 0.8
train_dataset = df.sample(frac=train_size, random_state=SEED).reset_index(drop=True)
val_dataset = df.drop(train_dataset.index).reset_index(drop=True)

print("FULL Dataset: {}".format(df.shape))
print("TRAIN Dataset: {}".format(train_dataset.shape))
print("TEST Dataset: {}".format(val_dataset.shape))

training_set = CustomDataset(train_dataset, tokenizer, MAX_LEN, SUMMARY_LEN)

# Defining the parameters for creation of dataloaders
train_params = {
    'batch_size': TRAIN_BATCH_SIZE,
    'shuffle': True,
    'num_workers': 0
}

training_loader = DataLoader(training_set, **train_params)

model = T5ForConditionalGeneration.from_pretrained("t5-base")
model = model.to(device)

optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

print('Initiating Fine-Tuning for the model on our dataset')

for epoch in range(TRAIN_EPOCHS):
    train(epoch, tokenizer, model, device, training_loader, optimizer)

model.save_pretrained("./weight")
tokenizer.save_pretrained('./weight')
