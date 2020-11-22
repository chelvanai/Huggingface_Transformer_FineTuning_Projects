import torch
from torch import cuda
from transformers import T5Tokenizer, T5ForConditionalGeneration

device = 'cuda' if cuda.is_available() else 'cpu'

tokenizer = T5Tokenizer.from_pretrained("./weight")
model = T5ForConditionalGeneration.from_pretrained('./weight', return_dict=True)
model = model.to(device)

ctext = "I am of course overjoyed to be here today in the role of ceremonial object. There is more than the usual " \
        "amount of satisfaction in receiving an honorary degree from the university that helped to form one’s " \
        "erstwhile callow and ignorant mind into the thing of dubious splendor that it is today; whose professors put " \
        "up with so many overdue term papers, and struggled to read one’s handwriting, of which ‘interesting’ is the " \
        "best that has been said; at which one failed to learn Anglo-Saxon and somehow missed Bibliography entirely, " \
        "a severe error which I trust no one present here today has committed; and at which one underwent " \
        "excruciating agonies not only of soul but of body, later traced to having drunk too much coffee in the " \
        "bowels of Wymilwood. It is to Victoria College that I can attribute the fact that Bell Canada, " \
        "Oxford University Press and McClelland and Stewart all failed to hire me in the summer of ‘63, " \
        "on the grounds that I was a) overqualified and b) couldn’t type, thus producing in me that state of " \
        "joblessness, angst and cosmic depression which everyone knows is indispensable for novelists and poets, " \
        "although nobody has ever claimed the same for geologists, dentists or chartered accountants. It is also due " \
        "to Victoria College, incarnated in the person of Northrop Frye, that I didn’t run away to England to become " \
        "a waitress, live in a garret, write masterpieces and get tuberculosis. He thought I might have more spare " \
        "time for creation if I ran away to Boston, lived in a stupor, wrote footnotes and got anxiety attacks, " \
        "that is, if I went to Graduate School, and he was right. So, for all the benefits conferred upon me by my " \
        "Alma Mater, where they taught me that the truth would make me free but failed to warn me of the kind of " \
        "trouble I’d get into by trying to tell it - I remain duly grateful. "

MAX_LEN = 512
SUMMARY_LEN = 150

source = tokenizer.batch_encode_plus([ctext], max_length=MAX_LEN, pad_to_max_length=True,
                                     return_tensors='pt')

ids = source['input_ids'].to(device, dtype=torch.long)
mask = source['attention_mask'].to(device, dtype=torch.long)


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

# generated summary
"""
in the role of ceremonial object. It is due to Victoria College, incarnated in Northrop Frye, that I didn’t run 
away to Boston, live in a stupor, write footnotes and get anxiety attacks, that is, if I went to Graduate School. 
Notably, Bell Canada, Oxford University Press, McClelland and Stewart all failed to hire me in the summer of ‘63, 
on the grounds that I was overqualified and couldn’t type.
"""