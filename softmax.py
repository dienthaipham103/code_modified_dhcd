from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
import keras

class SoftMax():
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self):
        model = Sequential()
        model.add(Dense(1024, activation='relu', input_shape=self.input_shape))
        model.add(Dropout(0.5))
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation='softmax'))

        # optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        optimizer = Adam()
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=optimizer,
                      metrics=['accuracy'])
        return model




# # Softmax regressor to classify images based on encoding
# classifier_model=Sequential()
# classifier_model.add(Dense(units=100,input_dim=x_train.shape[1],kernel_initializer='glorot_uniform'))
# classifier_model.add(BatchNormalization())
# classifier_model.add(Activation('tanh'))
# classifier_model.add(Dropout(0.3))
# classifier_model.add(Dense(units=10,kernel_initializer='glorot_uniform'))
# classifier_model.add(BatchNormalization())
# classifier_model.add(Activation('tanh'))
# classifier_model.add(Dropout(0.2))
# classifier_model.add(Dense(units=6,kernel_initializer='he_uniform'))
# classifier_model.add(Activation('softmax'))
# classifier_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),optimizer='nadam',metrics=['accuracy'])
