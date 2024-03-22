from keras.models import Model
from keras.layers import (Input, Conv1D, MaxPooling1D, 
                          Flatten, Dense, Dropout, Lambda, 
                          BatchNormalization, concatenate)
from keras import backend as K
from tensorflow.keras.optimizers.legacy import Adam

import tensorflow as tf
import numpy as np

# Hyperparameters
decision_lr = 1e-4
siamese_lr = 1e-3

input_siamese_shape = (9216, 1)
input_decision_shape = (32, 1)
decision_shape = (32, 1)

epochs = 10
alpha = 0.03


def triplet_loss(y_true, y_pred):
    # y_pred contains the anchor, positive and negative embedding vectors
    total_lenght = y_pred.shape.as_list()[-1]
    
    # The embeddings are separated in the output
    anchor, positive, negative = y_pred[:, :total_lenght//3], y_pred[:, total_lenght//3:2*total_lenght//3], y_pred[:, 2*total_lenght//3:]
    
    # Calculate the distances between anchor-positive and anchor-negative
    pos_dist = K.sum(K.square(anchor - positive), axis=1)
    neg_dist = K.sum(K.square(anchor - negative), axis=1)
    
    # Subtract the two previous distances and add alpha.
    basic_loss = pos_dist - neg_dist + alpha
    loss = K.maximum(basic_loss, 0.0)
    return loss


def siamese(x_input_shape):
    x_input = Input(shape=x_input_shape, name='siamese_input')

    x = x_input

    # Each convolutional layer includes a kernel regularizer (L2-norm) with a value of 10−3.
    # Convolutional Layer 1
    x = Conv1D(filters=64, kernel_size=5, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3))(x)
    x = BatchNormalization()(x)

    # Convolutional Layer 2
    x = Conv1D(filters=128, kernel_size=5, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3))(x)
    x = BatchNormalization()(x)

    # Convolutional Layer 3
    x = Conv1D(filters=256, kernel_size=3, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3))(x)
    x = BatchNormalization()(x)

    # Max Pooling
    # down-sampling the input representation by taking the maximum value over a spatial window of a specified size 
    x = MaxPooling1D(pool_size=4)(x)

    # Dropout
    x = Dropout(0.5)(x)

    # Flatten
    x = Flatten()(x)

    # Dense Layer
    x = Dense(32, activation='relu')(x)

    # return
    siamese_model = Model(inputs=x_input, outputs=x, name='siamese')

    siamese_model.summary()

    return siamese_model
    

def euclidean_distance(vectors):
    (featsA, featsB) = vectors
    return featsA - featsB

def decision_network(x_input_shape):
    x_input = Input(shape=x_input_shape, name='decision_input')
    x = x_input

    x = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2=0.01))(x)
    x = BatchNormalization()(x)

    x = Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2=0.01))(x)
    x = BatchNormalization()(x)

    x = Dropout(0.25)(x)

    x = Dense(1, activation='sigmoid')(x)
    decision_network = Model(inputs=x_input, outputs=x, name='decision_network')

    return decision_network
    

def train_siamese(data):
    # The training process is the following:
        # 1. Train the Siamese Network
        # 2. 
            # 2.1 First, the weights of the Siamese network are frozen (it is used as a feature extractor)
            # 2.2 then fully connected decision network is appended to the Siamese network 
                # and trained using a binary crossentropy loss function to optimize the network’s weights
    
    anchor, positive, negative = data

    # 1. The base network of siamese
    siamese_model = siamese(input_siamese_shape)

    # Define the tensors for the three input images
    input_anchor = Input(shape=input_siamese_shape, name='anchor_input')
    input_positive = Input(shape=input_siamese_shape, name='positive_input')
    input_negative = Input(shape=input_siamese_shape, name='negative_input')

    # Connect the inputs to the Siamese Network
    output_anchor = siamese_model(input_anchor)
    output_positive = siamese_model(input_positive)
    output_negative = siamese_model(input_negative)

    # Combine the outputs into a single tensor
    siamese_output = concatenate([output_anchor, output_positive, output_negative], axis=1)
    print("siamese_output: ", siamese_output.shape)

    # Optimizer and compile
    model = Model(inputs=[input_anchor, input_positive, input_negative], outputs=siamese_output, name='siamese_training_model')
    model.compile(loss=triplet_loss,
                    optimizer=Adam(learning_rate=siamese_lr),
                    metrics=['accuracy'])
    
    # Model summary
    model.summary()

    # Training
    model.fit(x=[anchor, positive, negative], y=np.zeros((anchor.shape[0], 1)), epochs=epochs)

    # Freeze the weights of the Siamese network are frozen (it is used as a feature extractor)
    siamese_model.trainable = False

    return siamese_model

def build_model(data):
    # trained siamese
    siamese = train_siamese(data)

    # define new siamese
    anchor_input = Input(shape=input_siamese_shape, name='anchor_input')
    target_input = Input(shape=input_siamese_shape, name='target_input')

    # Connect the inputs to the Siamese Network
    anchor_output = siamese(anchor_input)
    target_output = siamese(target_input)

    # distance
    distance = Lambda(euclidean_distance)([anchor_output, target_output])

    # decision network
    judge = decision_network(decision_shape)
    decision = judge(distance)

    # Model
    model = Model(inputs=[anchor_input, target_input], outputs=decision, name='model')

    optimizer = Adam(learning_rate=decision_lr)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    model.summary()

    return model

def train(data, target):
    # build model
    model = build_model(data)

    anchor, positive, negative = data

    # train
    model.fit([anchor, target], epochs=epochs)

    return model

def main(data, target):
    model = build_model(data)

    anchor, positive, negative = data

    prob = model.predict([anchor, target])
    threshold = 0.5
    decision = (prob > threshold).astype(int)

    print("prob: ", prob)
    print("decision: ", decision)

if __name__ == "__main__":
    anchor = np.random.rand(9216, 1)
    target = np.random.rand(9216, 1)
    positive = np.random.rand(9216, 1)
    negative = np.random.rand(9216, 1)

    anchor = anchor.reshape(1, 9216, 1)
    target = target.reshape(1, 9216, 1)
    positive = positive.reshape(1, 9216, 1)
    negative = negative.reshape(1, 9216, 1)

    data = (anchor, positive, negative)
    main(data, target)