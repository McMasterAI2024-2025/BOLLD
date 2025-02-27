import os
import sys
import tensorflow as tf
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from tensorflow.keras.layers import Conv3D, Dropout, BatchNormalization, Activation, MaxPool3D, LSTM, Dense, TimeDistributed, Bidirectional, Flatten, Reshape
from tensorflow.keras.models import Sequential
from utils import char_to_num

def create_lstm_layer(units, name):
    """Helper function to create LSTM layer with proper configuration"""
    return LSTM(
        units=units,
        kernel_initializer='orthogonal',
        return_sequences=True,
        name=name
    )

def load_model() -> Sequential:
    # Reset Keras session
    tf.keras.backend.clear_session()
    
    try:
        # Create model
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
        
        # Flatten and reshape
        model.add(TimeDistributed(Flatten()))
        model.add(Reshape((75, -1)))
        
        # Create LSTM layers with explicit scoping
        with tf.name_scope('bidirectional_1'):
            model.add(Bidirectional(create_lstm_layer(128, 'lstm_1')))
        model.add(Dropout(0.5))
        
        with tf.name_scope('bidirectional_2'):
            model.add(Bidirectional(create_lstm_layer(128, 'lstm_2')))
        model.add(Dropout(0.5))
        
        # Output layer
        model.add(Dense(
            char_to_num.vocabulary_size() + 1,
            kernel_initializer='he_normal',
            activation='softmax'
        ))
        
        # Build model
        model.build(input_shape=(None, 75, 46, 140, 1))
        
        # Compile model
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
        
    except Exception as e:
        print(f"Error creating model: {str(e)}")
        return None