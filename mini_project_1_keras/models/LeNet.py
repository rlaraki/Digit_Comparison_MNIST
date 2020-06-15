import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Activation
from keras import activations


class LeNet(tf.keras.Model):
    
    def __init__(self, d=0, only_image=False):
        super(LeNet, self).__init__()
        
        input_shape = 1 if only_image else 2
        output_shape = 10 if only_image else 2
        
        self.cnn_layers = Sequential([
                                      Conv2D(6, (2,2), data_format = 'channels_first', input_shape=(input_shape, 14, 14) , activation='relu'),
                                      MaxPooling2D(pool_size=(2,2), strides = (2,2)),
                                      Conv2D(7, (2,2), input_shape=(6, 6, 6), data_format = 'channels_first',activation='relu'),
                                      MaxPooling2D(pool_size=(2,2), strides = (2,2)),
                                      Conv2D(120, (2,2), input_shape=(7, 2, 2), data_format = 'channels_first',activation='relu')])
        
        
        self.fully_connected = Sequential([Flatten(), 
                                           Dense(84, activation='relu', input_shape=(120, )), 
                                           Dense(output_shape, activation='softmax')])
        
    
    def call(self, inputs):
        output = self.cnn_layers(inputs)
        output = self.fully_connected(output)
        return output
            