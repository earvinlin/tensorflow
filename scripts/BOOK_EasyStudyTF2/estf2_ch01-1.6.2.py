"""
§1.6.2 Sequential Model
"""
#import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras import layers

import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras import layers
from keras import layers

#from tensorflow.keras.utils import plot_model
from keras.utils import plot_model
from IPython.display import Image

# Method 1
model = keras.Sequential(name='Sequential')
model.add(layers.Dense(64, activation='relu', input_shape=(784,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Methon 2

plot_model(model, to_file='Functional_API_Sequential_model.png')

Image('Functional_API_Sequential_Model.png')

