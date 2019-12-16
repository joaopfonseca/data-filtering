import os
from sklearn.base import BaseEstimator
import numpy as np
from keras.layers import (
    Input,
    Dense,
    BatchNormalization,
    Concatenate
)
from keras.models import Model
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

class MLPAutoEncoder:
    """
    """
    def __init__(self, num_bands, filepath='best_model.hdf5'):
        self.num_bands = num_bands

        self._decoder(self._encoder(num_bands))
        self.model = Model(inputs=self.input_layer, outputs=self.decoder_output)
        self.model.compile(loss='mean_squared_error', optimizer=RMSprop())#, metrics=['accuracy'])
        #self.model.summary()
        abspath = os.path.abspath('.')
        if filepath:
            self.filepath = os.path.abspath(os.path.join(abspath,filepath))
            checkpoint = ModelCheckpoint(self.filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
            self.callbacks_list = [checkpoint]
        else:
            self.callbacks_list = None

    def _encoder(self, num_features):
        """
        """
        self.num_features = num_features
        self.input_layer = Input((num_features,))
        layer1 = Dense(32, input_shape=self.input_layer._keras_shape, activation='relu')(self.input_layer)
        layer1 = BatchNormalization()(layer1)
        layer2 = Dense(16, activation='relu')(layer1)
        layer2 = BatchNormalization()(layer2)
        layer3 = Dense(4, activation='relu')(layer2)
        self.encoder_output = BatchNormalization()(layer3)
        return self.encoder_output

    def _decoder(self, encoder_output):
        """
        """
        layer4 = Dense(16, input_shape=self.encoder_output._keras_shape, activation='relu')(encoder_output)
        layer4 = BatchNormalization()(layer4)
        layer5 = Dense(32, activation='relu')(layer4)
        layer5 = BatchNormalization()(layer5)
        self.decoder_output = Dense(self.num_features, activation=None)(layer5)
        return self.decoder_output

    def load_weights(self, filepath):
        self.filepath = filepath
        self.model = load_model(filepath)
        self.model.compile(loss='mean_squared_error', optimizer=RMSprop())

    def fit(self, X, y, batch_size=256, epochs=100):
        # transform matrices to correct format
        self.num_bands = X.shape[-1]
        X = X.reshape(-1, self.num_bands,)
        y = y.reshape(-1, self.num_bands,)

        self.history = self.model.fit(
            x=X,
            y=y,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=self.callbacks_list
        )

    def predict(self, X, filepath=None):
        # assert: self.filepath or filepath must exist
        if filepath:
            self.load_weights(filepath)
            self.model.compile(loss='mean_squared_error', optimizer=RMSprop())
        #else:
        #    self.load_model(self.filepath)
        #self.model.compile(loss='categorical_crossentropy', optimizer=self.adam, metrics=['accuracy'])

        X_pred = self.model.predict(X)
        mse = ((X_pred-X)**2).mean(axis=1)
        return mse

class MLPEncoderClassifier:
    def __init__(self, encoder_list, num_targets, filepath='best_model.hdf5'):
        self.num_targets = num_targets
        self.num_encoders = len(encoder_list)

        MergedEncoders = Concatenate()([model.encoder_output for model in encoder_list])
        self._MLPClassifier(MergedEncoders)

        self.model = Model(inputs=[model.input_layer for model in encoder_list], outputs=self.output_layer)
        self.adam = Adam(lr=0.001, decay=1e-06)
        self.model.compile(loss='categorical_crossentropy', optimizer=self.adam, metrics=['accuracy'])
        self.model.summary()
        abspath = os.path.abspath('.')
        if filepath:
            self.filepath = os.path.abspath(os.path.join(abspath,filepath))
            checkpoint = ModelCheckpoint(self.filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
            self.callbacks_list = [checkpoint]
        else:
            self.callbacks_list = None

    def _MLPClassifier(self, merged_encoders_outputs):
        layer1 = BatchNormalization()(merged_encoders_outputs)
        layer1 = Dense(32, activation='relu')(layer1)
        layer1 = BatchNormalization()(layer1)
        layer2 = Dense(16, activation='relu')(layer1)
        layer2 = BatchNormalization()(layer2)
        self.output_layer = Dense(self.num_targets, activation='sigmoid')(layer2)
        return self.output_layer

    def fit(self, X, y, batch_size=256, epochs=100):
        # transform matrices to correct format
        self.num_bands = X.shape[-1]
        X = X.reshape(-1, self.num_bands,)
        y = np_utils.to_categorical(y, num_classes=self.num_targets)

        self.history = self.model.fit(
            x=[X for i in range(self.num_encoders)],
            y=y,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=self.callbacks_list
        )

    def predict(self, X, filepath=None):
        # assert: self.filepath or filepath must exist
        if filepath:
            self.load_weights(filepath)
            self.model.compile(loss='mean_squared_error', optimizer=RMSprop())
        #else:
        #    self.load_model(self.filepath)
        #self.model.compile(loss='categorical_crossentropy', optimizer=self.adam, metrics=['accuracy'])

        y_pred = np.argmax(self.model.predict([X for i in range(self.num_encoders)]), axis=1)
        return y_pred

class AEMLPClassifier(BaseEstimator):
    """
    TODO: custom definition of epochs and batch_size
    """
    def __init__(self, batch_size=32, epochs=100, filepath=None):
        super().__init__()
        self.batch_size = batch_size
        self.epochs = epochs
        self.filepath = filepath

    def fit(self, X, y):

        autoencoders = {}
        mses = {}
        for label in np.unique(y):
            autoencoder = MLPAutoEncoder(X.shape[-1], self.filepath)
            X_label = X[y==label]
            #print(f'Predicting label {label}...')
            autoencoder.fit(X_label, X_label, batch_size=self.batch_size, epochs=self.epochs)
            autoencoders[label] = autoencoder

        self.clf = MLPEncoderClassifier(autoencoders.values(), int(np.unique(y).max())+1, filepath=self.filepath)
        self.clf.fit(X, y, self.batch_size, self.epochs)
        return self

    def predict(self, X):
        return self.clf.predict(X)
