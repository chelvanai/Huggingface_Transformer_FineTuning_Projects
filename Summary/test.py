import torch
from torch import cuda
from transformers import T5Tokenizer, T5ForConditionalGeneration


class Summary:
    def __init__(self):
        self.device = 'cuda' if cuda.is_available() else 'cpu'

        self.tokenizer = T5Tokenizer.from_pretrained("./weight")
        self.model = T5ForConditionalGeneration.from_pretrained('./weight', return_dict=True)
        self.model = self.model.to(self.device)

    def getSummary(self, ctext):
        MAX_LEN = 512
        SUMMARY_LEN = 150

        source = self.tokenizer.batch_encode_plus([ctext], max_length=MAX_LEN, pad_to_max_length=True,
                                                  return_tensors='pt')

        ids = source['input_ids'].to(self.device, dtype=torch.long)
        mask = source['attention_mask'].to(self.device, dtype=torch.long)

        generated_ids = self.model.generate(
            input_ids=ids,
            attention_mask=mask,
            max_length=150,
            num_beams=2,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True
        )

        preds = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in
                 generated_ids]

        return preds


