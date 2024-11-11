from cnn_model import build_cnn
import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

#distributed strategy for multi-device setups
strategy = tf.distribute.MirroredStrategy()
with strategy.scope: 
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    train_images, test_images = train_images/255.0, test_images/255.0     #preprocessing data, normalizes pixel values, 

model = build_cnn((32,32,3), 10)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#callbacks for model checkpoints/early stopping
checkpoint_cb = ModelCheckpoint('cnn_best_model.h5', save_best_only=True)    #saves best model based on validation performance 
early_stopping_cb = EarlyStopping(patience=5, restore_best_weight=True)      #stops training if validation accuracy isn't improving afetr 5 epochs --> prevents overfitting

#modified to a HPC environment capacity  - i upgraded to larger batch sizes (128, we could do 256) & longer training epochs
history = model.fit(
    train_images, train_labels, 
    epochs = 20, 
    validation_data=(test_images, test_labels), 
    batch_size=128, 
    callbacks=[checkpoint_cb, early_stopping_cb])


