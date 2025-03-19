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


print("len=", len(decode_review))
print("type=", type(decode_review))



print("=== PGM END ===")

