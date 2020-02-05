#### LRFinder (LR)

LRFinder pour Keras

Variation exponentielle du Learning rate entre deux bornes (min, max)

```python
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Bidirectional, Flatten, Dense
from Utils.selfAttention import SelfAttentionLayer
from LRFinder import *

model = Sequential()
model.add(Embedding(len(tokenizer.word_index), 300, input_length = word_seq_train.shape[1], weights = [embedding_matrix]))
model.add(Bidirectional(LSTM(64, return_sequences = True)))
model.add(SelfAttention(300, 15, attention_regularizer_weights = 0.5, return_attention = False))
model.add(Flatten())
model.add(Dense(2))

model.compile(optimizer = "Adam", loss = "binary_crossentropy", metrics = ['accuracy'])

Lrfinder = LRFinder(model)
Lrfinder.find_sample_fit(X_train, y_train, min_lr = 0.0001, max_lr = 2, X_test, y_test, batch_size = 128, epochs = 1)

Lrfinder.plot()
```
