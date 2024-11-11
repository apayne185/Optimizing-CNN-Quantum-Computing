import tensorflow as tf
from qcnn_model import * 

#generates random qubit states for excitations and their labels
def generate_data(cluster_state_bits):
    excitations = np.random.rand(100,1) 
    labels = np.random.randint(0,2,size=(100,))
    return excitations, labels, excitations, labels         #for both train/test


# Generate some training data.
train_excitations, train_labels, test_excitations, test_labels = generate_data(cluster_state_bits)


# Custom accuracy metric.
@tf.function
def custom_accuracy(y_true, y_pred):
    y_true = tf.squeeze(y_true)
    y_pred = tf.map_fn(lambda x: 1.0 if x >= 0 else -1.0, y_pred)
    return tf.keras.backend.mean(tf.keras.backend.equal(y_true, y_pred))


qcnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.02),
                   loss="binary_crossentropy",     #because we are dealing with classification
                   metrics=[custom_accuracy])

history = qcnn_model.fit(x=train_excitations,
                         y=train_labels,
                         batch_size=16,
                         epochs=25,
                         verbose=1,
                         validation_data=(test_excitations, test_labels))