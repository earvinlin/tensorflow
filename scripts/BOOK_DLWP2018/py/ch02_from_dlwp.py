###--- § 2-1 加載Keras中的MNIST數據集 ---###
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print("len(train_labels): ", len(train_labels))
print("train_labels     : ", train_labels)
print("len(test_labels): ", len(test_labels))
print("test_labels     : ", test_labels)

###--- § 2-2 網路架構 ---###
from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
network.add(layers.Dense(10, activation='softmax'))

###--- § 2-3 編譯步驟 ---###
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

###--- § 2-4 准備圖像數據 ---###
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((60000, 28 * 28))
test_images = test_images.astype('float32') / 255

###--- § 2-5 准備標籤 ---###
from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)












