# -*- coding: utf-8 -*-
"""
Created on Sun Jul 02 14:22:45 2023
 
@author: eacosta
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from keras.callbacks import CSVLogger
import matplotlib.pyplot as plt
import time
from config import LOG_FILE, LOG_DIR, QRB_REP
from helpers import qcprint

class ANN():
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    
    model = Sequential()
    
    def __init__(self, train_data, train_labels, test_data, test_labels):
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
    
        # Build the RNN model
        self.model = Sequential([
            LSTM(units=50, activation='relu', input_shape=(QRB_REP, 1)),
            Dense(units=1, activation='sigmoid')
        ])
    
    def train(self, epochs):
        qcprint("\nTRAINING ANN")
        start_time = time.time()

        # Compile the model
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        csv_logger = CSVLogger(LOG_FILE, append=True, separator=';')

        history = self.model.fit(self.train_data, self.train_labels, epochs=epochs, 
                                 batch_size=32, callbacks=[csv_logger], verbose=0,
                                 validation_data=(self.test_data, self.test_labels))
        qcprint("Training ANN took %s sec" % (time.time() - start_time))
        
        # Plot training loss
        plt.plot(history.history['loss'], label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('ANN Training and Validation Loss')
        plt.savefig(LOG_DIR+"ann_cost_{}.png".format(time.time()))
        
    def validate(self):
        # Evaluate the model
        loss, accuracy = self.model.evaluate(self.test_data, self.test_labels,
                                             verbose=0)
        qcprint(f"Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}")
        
        return accuracy
