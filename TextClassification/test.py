from transformers import DistilBertTokenizer
from transformers import TFDistilBertForSequenceClassification
import tensorflow as tf
import pandas as pd

loaded_tokenizer = DistilBertTokenizer.from_pretrained('./weight')
loaded_model = TFDistilBertForSequenceClassification.from_pretrained('./weight')


# This sentence is in 13861 row in CSV file, The manual annotation is [H:N	H:H]

test_text = "\"Oh, how nicely it is made,\" exclaimed the ladies."

predict_input = loaded_tokenizer.encode(test_text,
                                        truncation=True,
                                        padding=True,
                                        return_tensors="tf")

print("")

output = loaded_model(predict_input)[0]
print(output)
prediction_value = tf.argmax(output, axis=1).numpy()[0]
print(prediction_value)

df = pd.read_csv('EncodedOutput.csv')
emotion = df[df['EncodedValues'] == prediction_value]['Emotion'].values

print(emotion)
