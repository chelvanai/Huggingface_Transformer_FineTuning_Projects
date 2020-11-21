from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

tokenizer = AutoTokenizer.from_pretrained("./weight")

model = AutoModelForQuestionAnswering.from_pretrained("./weight")

text =  """He said he is very tired, do not disturb me """

question = "what did he say"

inputs = tokenizer.encode_plus(question, text, add_special_tokens=True, return_tensors="pt")
input_ids = inputs["input_ids"].tolist()[0]

text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
answer_start_scores, answer_end_scores = model(**inputs)

answer_start = torch.argmax(
    answer_start_scores
)  # Get the most likely beginning of answer with the argmax of the score
answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score

answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

print(f"Question: {question}")
print(f"Answer: {answer}\n")
