import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn(in_shape, n_classes): 
    model = models.Sequential()         #allows for linear stacking layers 

    #convolutional layer across inputs 
    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape= (in_shape)))  #32 filters, 3x3 kernel, inputshape=channels (RGB)
    model.add(layers.MaxPooling2D((2,2))) #downsampling layer for reducing spatial dim of feature maps provided by Conv2D layer above
    model.add(layers.Conv2D(64, (3,3), activation='relu'))  #ReLU (rectified linear unit) replaces all - values with 0 for non-linearity
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    #every Conv2D and MaxPool layer outputs a 2d tensor (height, width, channels)


    #feeds ouput tensor into Dense layers for classification
    model.add(layers.Flatten())  #turns 2d vector into 1d 
    model.add(layers.Dense(128, activation='relu'))  #128 layers , learns complex features
    model.add(layers.Dense(n_classes, activation='softmax')) #allows for multi-class probability output

    return model

