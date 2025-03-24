from tensorflow import keras
from keras.datasets import reuters

print("=== PGM START ===")

# 3-12 加載路透社數據集
(train_data, train_labels), (test_data, test_labels) = \
    reuters.load_data(num_words=10000)

print("train_data's size= ", len(train_data))
print("test_data's size= ", len(test_data))
print("train_data[0]= ", train_data[10])

# 3-13 將索引解碼為新聞文本
word_index = reuters.get_word_index()
reverse_word_index = \
    dict([(value, key) for (key, value) in word_index.items()])
decoded_newswire = \
    ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])

print("train_labels[10]= ", train_labels[10])

# 3-14 編碼數據 (數據向量化)
import numpy as np
def vectorize_sequences(sequences, dimension=10000) :
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences) :
        results[i, sequence] = 1.
    return results

x_train = vectorize_sequences(train_data) # 將訓練數據向量化
x_test = vectorize_sequcences(test_data)  # 將測試數據向量化









print("=== PGM END ===")

