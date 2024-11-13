import tensorflow as tf
from tensorflow.keras import layers, models
from quantum_cnn.qcnn_layers import QuantumConvLayer
import yaml

with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

cnn_filters = config['cnn_filters']
cnn_kernel = config['cnn_kernel']
cnn_activation = config['cnn_activation']
output_activation = config['output_activation']

#this cnn includes QC as preprocessing 
def build_cnn(in_shape, n_classes, bits, symbols): 
    model = models.Sequential()         #allows for linear stacking layers 

    #apply QCL as preprocessing step
    model.add(QuantumConvLayer(bits=bits, symbols=symbols, input_shape= in_shape))

    #classical convolutional layer across inputs 
    model.add(layers.Conv2D(cnn_filters[0], cnn_kernel, activation=cnn_activation, input_shape= in_shape))  #32 filters, 3x3 kernel, inputshape=channels (RGB)
    model.add(layers.MaxPooling2D((2,2))) #downsampling layer for reducing spatial dim of feature maps provided by Conv2D layer above
    model.add(layers.Conv2D(cnn_filters[1], cnn_kernel, activation=cnn_activation))  #ReLU (rectified linear unit) replaces all - values with 0 for non-linearity
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(cnn_filters[2], cnn_kernel, activation=cnn_activation))
    #every Conv2D and MaxPool layer outputs a 2d tensor (height, width, channels)


    #feeds ouput tensor into Dense layers for classification
    model.add(layers.Flatten())  #turns 2d vector into 1d 
    model.add(layers.Dense(cnn_filters[3], activation=cnn_activation))  #256 layers , learns complex features
    model.add(layers.Dense(n_classes, activation=output_activation)) #allows for multi-class probability output

    return model



#this one does not
def build_classic_cnn(in_shape, n_classes): 
    model = models.Sequential()  

    model.add(layers.Conv2D(cnn_filters[0], cnn_kernel, activation=cnn_activation, input_shape=in_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(cnn_filters[1], cnn_kernel, activation=cnn_activation))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(cnn_filters[2], cnn_kernel, activation=cnn_activation))

    model.add(layers.Flatten())
    model.add(layers.Dense(cnn_filters[3], activation=cnn_activation))
    model.add(layers.Dense(n_classes, activation=output_activation))

    return model
