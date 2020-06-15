from tensorflow import keras
import tensorflow as tf
import numpy as np

def mnist_to_pairs(nb, input, target):
    
    input = keras.layers.AveragePooling2D((2,2))(input)
    input = tf.reshape(input, [-1, 1, 14, 14])
    a = np.random.permutation(input.shape[0])
    a = a[:2 * nb].reshape(nb, 2)
    input = tf.keras.backend.concatenate((tf.gather(input,a[:, 0] , axis=0), tf.gather(input,a[:, 1] , axis=0)), axis=1)
    classes = tf.gather(target, a, axis = 0)
    target = (classes[:, 0] <= classes[:, 1])
    target = tf.keras.backend.cast(target, dtype= 'float32')
    
    return input, target, classes
    
######################################################################

def generate_pair_sets(nb):
  
    (train_input, train_target), (test_input, test_target) = tf.keras.datasets.mnist.load_data(
    path='mnist.npz')
    
    train_input = train_input.reshape(-1, 28, 28, 1).astype('float32')
    
    test_input = test_input.reshape(-1, 28, 28, 1).astype('float32')
    

    return mnist_to_pairs(nb, train_input, train_target) + \
           mnist_to_pairs(nb, test_input, test_target)