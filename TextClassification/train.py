import pandas as pd
from transformers import DistilBertTokenizer
from transformers import TFDistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import tensorflow as tf
import csv

df = pd.read_csv('sentiment_data.csv')

print(df.head())

label_encoder = preprocessing.LabelEncoder()
df['Label'] = label_encoder.fit_transform(df['Label'])

data_texts = df["Sentence"].to_list()
data_labels = df["Label"].to_list()

# Write Encoded value into CSV file
mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print(mapping)

w = csv.writer(open("EncodedOutput.csv", "w"))
w.writerow(['Emotion','EncodedValues'])
for key, val in mapping.items():
    w.writerow([key, val])


# Split Train and Validation data
train_texts, val_texts, train_labels, val_labels = train_test_split(data_texts, data_labels, test_size=0.2, random_state=0)

# Keep some data for inference (testing)
train_texts, test_texts, train_labels, test_labels = train_test_split(train_texts, train_labels, test_size=0.01, random_state=0)

# Tokenizing the text
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)

# Creating a Dataset object for Tensorflow
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    train_labels
))
val_dataset = tf.data.Dataset.from_tensor_slices((
    dict(val_encodings),
    val_labels
))

# Fine-tuning Option : Using native Tensorflow
model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=6)

optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
model.compile(optimizer=optimizer, loss=model.compute_loss, metrics=['accuracy'])


# Training (Fine-Tuning)
model.fit(train_dataset.shuffle(1000).batch(16), epochs=10, batch_size=16,
          validation_data=val_dataset.shuffle(1000).batch(16))

# Save model
save_directory = "./weight" # change this to your preferred location

model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)
