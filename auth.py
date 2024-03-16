from keras.models import Model
from keras.layers import (Input, Conv1D, MaxPooling1D, 
                          Flatten, Dense, Dropout, Lambda, BatchNormalization)
from keras import backend as K

import tensorflow as tf
import numpy as np

def siamese(x_input):
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
    return x

def euclidean_distance(vectors):
    (featsA, featsB) = vectors
    # sumSquared = K.sum(K.square(featsA - featsB), axis=1, keepdims=True)
    # print("sumSquared.shape: ", sumSquared.shape)
    # return K.sqrt(K.maximum(sumSquared, K.epsilon()))
    return featsA - featsB


def decision_network(x):
    x = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2=0.01))(x)
    x = BatchNormalization()(x)

    x = Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2=0.01))(x)
    x = BatchNormalization()(x)

    x = Dropout(0.25)(x)

    x = Dense(1, activation='sigmoid')(x)

    return x

def build_model(anchor, target):
    inputA = Input(shape=(anchor.shape[0], anchor.shape[1]))

    featsA = siamese(inputA)

    inputB = Input(shape=(target.shape[0], target.shape[1]))
    
    featsB = siamese(inputB)

    print("featsA.shape: ", featsA.shape, "featsB.shape: ", featsB.shape)

    distance = Lambda(euclidean_distance)([featsA, featsB])

    decision = decision_network(distance)

    model = Model(inputs=[inputA, inputB], outputs=decision)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def main(anchor, target):
    model = build_model(anchor, target)
    model.summary()

    prob = model.predict([anchor, target])
    threshold = 0.5
    decision = (prob > threshold).astype(int)

    print("prob: ", prob)
    print("decision: ", decision)

if __name__ == "__main__":
    anchor = np.random.rand(9216, 1)
    target = np.random.rand(9216, 1)
    anchor = anchor.reshape(anchor.shape[0], anchor.shape[1], 1)
    target = target.reshape(target.shape[0], target.shape[1], 1)
    print("anchor.shape:", anchor.shape, "target.shape:", target.shape)
    main(anchor, target)