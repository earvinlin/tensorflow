from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from keras.datasets import imdb

print("=== PGM START ===")

(train_data, train_labels), (test_data, test_labels) = \
    imdb.load_data(num_words=10000)

print("train_data[]=", train_data[0])

word_index = imdb.get_word_index()

#print(word_index['this'])

reverse_word_index = dict(
        [(value, key) for (key, value) in word_index.items()])
# 「?」: 指定值不存在時，返回此值
decode_review = ' '.join(
        [reverse_word_index.get(i-3, '?') for i in train_data[0]])
print("decode_review=", decode_review)


#print("len=", len(decode_review))
#print("type=", type(decode_review))

# 3-2 
import numpy as np
def vectorize_sequences(sequences, dimension=10000) :
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences) :
        results[i, sequence] = 1.
    return results

x_train = vectorize_sequences(train_data) # 訓練數據向量化
x_test = vectorize_sequences(test_data) # 測試數據向量化

print(x_train[0])

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')





print("=== PGM END ===")

