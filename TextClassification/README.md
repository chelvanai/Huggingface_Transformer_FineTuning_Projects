**Train**

put your dataset instead for sentiment.csv here 6 emotion given as training hyper parameter
so if you want to configure as your dataset.

>>> model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=6) 
 
 change num_label of your one

>>> python train.py

**Test**

>>> python test.py

 
