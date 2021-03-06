from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from torch import cuda
import pandas as pd

device = 'cuda' if cuda.is_available() else 'cpu'

tokenizer = T5Tokenizer.from_pretrained("./weight")
model = T5ForConditionalGeneration.from_pretrained('./weight', return_dict=True)
model = model.to(device)

ctext = "Jaffna is the capital city of the Northern Province of Sri Lanka. It is the administrative headquarters of " \
        "the Jaffna District located on a peninsula of the same name. With a population of 88,138 in 2012, " \
        "Jaffna is Sri Lanka's 12th most populous city. Jaffna is approximately six miles from Kandarodai which " \
        "served as an emporium in the Jaffna peninsula from classical antiquity. Jaffna's suburb Nallur, " \
        "served as the capital of the four-century-long medieval Jaffna Kingdom. Prior to the Sri Lankan Civil War, " \
        "it was Sri Lanka's second most populous city after Colombo. The 1980s insurgent uprising led to extensive " \
        "damage, expulsion of part of the population, and military occupation. Since the end of civil war in 2009, " \
        "refugees and internally displaced people began returning to homes, while government and private sector " \
        "reconstruction started taking place. Historically, Jaffna has been a contested city "

VALID_BATCH_SIZE = 2
MAX_LEN = 512
SUMMARY_LEN = 150

data = {'ctext': [ctext]}

val_dataset = pd.DataFrame(data, columns=['ctext'])

print(val_dataset.head())


class CustomDataset1(Dataset):

    def __init__(self, dataframe, tokenizer, source_len, summ_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = summ_len
        self.ctext = self.data.ctext

    def __len__(self):
        return len(self.ctext)

    def __getitem__(self, index):
        ctext = str(self.ctext[index])
        ctext = ' '.join(ctext.split())

        source = self.tokenizer.batch_encode_plus([ctext], max_length=self.source_len, pad_to_max_length=True,
                                                  return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long),

        }


val_params = {
    'batch_size': VALID_BATCH_SIZE,
    'shuffle': False,
    'num_workers': 0
}

val_set = CustomDataset1(val_dataset, tokenizer, MAX_LEN, SUMMARY_LEN)
val_loader1 = DataLoader(val_set, **val_params)

model.eval()

with torch.no_grad():
    for _, data in enumerate(val_loader1, 0):
        ids = data['source_ids'].to(device, dtype=torch.long)
        mask = data['source_mask'].to(device, dtype=torch.long)

        print(ids.shape)
        print(mask.shape)

        generated_ids = model.generate(
            input_ids=ids,
            attention_mask=mask,
            max_length=150,
            num_beams=2,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True
        )

        preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in
                 generated_ids]

print(preds)
