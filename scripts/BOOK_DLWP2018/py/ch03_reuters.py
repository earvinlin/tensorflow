from tensorflow import keras
from keras.datasets import reuters

# 3-12
(train_data, train_labels), (test_data, test_labels) = \
    reuters.load_data(num_words=10000)

print("=== PGM START ===")

print("len(train_data)= ", len(train_data))
print("len(test_data)= ", len(test_data))
print("train_data[10]= ", train_data[10])

# 3-13
word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) \
    for (key, value) in word_index_items()])
decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') 
    for i in train_data[0]])

print("train_labels[10]= ", train_labels[10])



























print("=== PGM END ===")

