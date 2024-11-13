from cnn_model import *
from quantum_cnn.qcnn_model import * 
import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import cirq
import sympy
from qisket import IBMQ
from qisket_imb_runtime import Session, Options
import yaml

#IBM integration for training cnn
#IBMQ.save_account('APITOKEN', overwrite=True)
#options = Options(shots=1024)
#session = Session(backend='ibmq_qasm_simulator', options=options)


with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)



#distributed strategy for multi-device setups
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: ', strategy.num_replicas_in_sync)

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images/255.0, test_images/255.0     #preprocessing data, normalizes pixel values, 

train_w_qcnn = True   #set to false to train classical

with strategy.scope: 
    if train_w_qcnn:
        bits=[cirq.GridQubit(0,i) for i in range(config['quantum_layers'])]
        target_state = [1,0,1,0]    #this gets modified
        symbols = sympy.symbols('x0:12')
        model = build_cnn(bits, target_state, symbols, num_classes= 10)
    else:
        model = build_classic_cnn(input_shape=(32,32,3))

    model.compile(optimizer=config['optimizer'], loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#callbacks for model checkpoints/early stopping
checkpoint_cb = ModelCheckpoint(config['output_directory'] + '/cnn_best_model.h5', save_best_only=True)    #saves best model based on validation performance 
early_stopping_cb = EarlyStopping(patience=5, restore_best_weight=True)      #stops training if validation accuracy isn't improving afetr 5 epochs --> prevents overfitting

#modified to a HPC environment capacity  - i upgraded to larger batch sizes (128, we could do 256) & longer training epochs
history = model.fit(
    train_images, train_labels, 
    epochs = config['epochs'], 
    validation_data=(test_images, test_labels), 
    batch_size= config['batch_size'], 
    callbacks=[checkpoint_cb, early_stopping_cb])


