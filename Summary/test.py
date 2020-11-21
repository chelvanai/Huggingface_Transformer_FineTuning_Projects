from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from torch import cuda
import pandas as pd

tokenizer = AutoTokenizer.from_pretrained("./weight")
model = T5ForConditionalGeneration.from_pretrained("./weight")

device = 'cuda' if cuda.is_available() else 'cpu'

ctext = "The Transformer is a deep learning model introduced in 2017, used primarily in the field of natural " \
        "language processing. Like recurrent neural networks, Transformers are designed to handle sequential data, " \
        "such as natural language, for tasks such as translation and text summarization. However, unlike RNNs, " \
        "Transformers do not require that the sequential data be processed in order. For example, if the input data " \
        "is a natural language sentence, the Transformer does not need to process the beginning of it before the " \
        "end. Due to this feature, the Transformer allows for much more parallelization than RNNs and therefore " \
        "reduced training times. Since their introduction, Transformers have become the model of choice for " \
        "tackling many problems in NLP, replacing older recurrent neural network models such as the long short-term " \
        "memory. Since the Transformer model facilitates more parallelization during training, it has enabled " \
        "training on larger datasets than was possible before it was introduced. "

VALID_BATCH_SIZE = 2  # input batch size for testing (default: 1000)
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
