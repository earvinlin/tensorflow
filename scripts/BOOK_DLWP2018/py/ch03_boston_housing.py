# 3-24 加載波士頓房價數據
from tensorflow import keras
from keras.datasets import boston_housing

print("=== PGM START ===")

(train_data, train_targets), (test_data, test_targets) = \
    boston_housing.load_data()

print("train_data.shape= ", train_data.shape)
print("test_data.spahe = ", test_data.shape)
print("train_targets= ", train_targets)

















print("=== PGM END ===")

