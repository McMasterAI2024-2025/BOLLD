import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from tensorflow.keras.layers import Conv3D, Dropout, BatchNormalization, Activation, MaxPool3D, LSTM, Dense, TimeDistributed, Bidirectional, Flatten, Reshape
from tensorflow.keras.models import Sequential
from utils import char_to_num

def load_model() -> Sequential:
    model = Sequential()

    # First Conv3D block
    model.add(Conv3D(128, 3, input_shape=(75, 46, 140, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool3D(pool_size=(1, 2, 2)))

    # Second Conv3D block
    model.add(Conv3D(256, 3, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool3D(pool_size=(1, 2, 2)))

    # Third Conv3D block
    model.add(Conv3D(75, 3, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool3D(pool_size=(1, 2, 2)))

    # Flatten spatial dimensions but keep time dimension
    model.add(TimeDistributed(Flatten()))

    # Reshape to match LSTM input format (time, features)
    model.add(Reshape((75, -1)))

    # print("Output shape before LSTM:", model.output_shape)

    # First LSTM block
    model.add(Bidirectional(LSTM(128, kernel_initializer='orthogonal', return_sequences=True)))


    model.add(Dropout(0.5))

    # Second LSTM block
    model.add(Bidirectional(LSTM(128, kernel_initializer='orthogonal', return_sequences=True)))
    model.add(Dropout(0.5))

    # Fully connected output layer
    model.add(Dense(char_to_num.vocabulary_size() + 1, kernel_initializer='he_normal', activation='softmax'))

    # Load the weights if they exist
    current_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(current_dir, 'models', 'new_best_weights2.weights.h5')
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
    else:
        print(f"Warning: Weights file not found at {weights_path}")

    return model
