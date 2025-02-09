import os
import sys
import tensorflow as tf
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from tensorflow.keras.layers import Conv3D, Dropout, BatchNormalization, Activation, MaxPool3D, LSTM, Dense, TimeDistributed, Bidirectional, Flatten, Reshape
from tensorflow.keras.models import Sequential
from utils import char_to_num

def load_model() -> Sequential:
    # Clear any existing backend session
    tf.keras.backend.clear_session()
    
    # Build model with explicit names for layers
    model = Sequential([
        # First Conv3D block
        Conv3D(128, 3, input_shape=(75, 46, 140, 1), padding='same', name='conv3d_1'),
        BatchNormalization(name='bn_1'),
        Activation('relu', name='relu_1'),
        MaxPool3D(pool_size=(1, 2, 2), name='maxpool3d_1'),

        # Second Conv3D block
        Conv3D(256, 3, padding='same', name='conv3d_2'),
        BatchNormalization(name='bn_2'),
        Activation('relu', name='relu_2'),
        MaxPool3D(pool_size=(1, 2, 2), name='maxpool3d_2'),

        # Third Conv3D block
        Conv3D(75, 3, padding='same', name='conv3d_3'),
        BatchNormalization(name='bn_3'),
        Activation('relu', name='relu_3'),
        MaxPool3D(pool_size=(1, 2, 2), name='maxpool3d_3'),

        # Flatten and reshape
        TimeDistributed(Flatten(), name='time_distributed_flatten'),
        Reshape((75, -1), name='reshape_1'),

        # LSTM blocks
        Bidirectional(LSTM(128, kernel_initializer='orthogonal', return_sequences=True), 
                     name='bidirectional_1'),
        Dropout(0.5, name='dropout_1'),

        Bidirectional(LSTM(128, kernel_initializer='orthogonal', return_sequences=True),
                     name='bidirectional_2'),
        Dropout(0.5, name='dropout_2'),

        # Output layer
        Dense(char_to_num.vocabulary_size() + 1, 
              kernel_initializer='he_normal', 
              activation='softmax',
              name='output')
    ])

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy']
    )

    # Load weights
    current_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(current_dir, 'models', 'new_best_weights2.weights.h5')
    if os.path.exists(weights_path):
        try:
            model.load_weights(weights_path)
            print(f"Successfully loaded weights from {weights_path}")
        except Exception as e:
            print(f"Error loading weights: {str(e)}")
    else:
        print(f"Warning: Weights file not found at {weights_path}")

    return model