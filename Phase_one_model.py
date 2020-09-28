from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Dense, Flatten

class Modle_phase_one:
    @staticmethod
    def build_model(width, height, depth, classes):
        #Initialize he model
        model = Sequential()
        input_shape = (height,width, depth)
        activ = 'relu'
        kernel_size = (5,5)

        #First Convolutional layer:
        model.add(Conv2D(30, kernel_size = kernel_size, padding='same', input_shape=input_shape))
        model.add(Activation(activation=activ))
        model.add(MaxPooling2D(pool_size=(2,2), strides= (2,2)))

        #Second Convolutional layer:
        model.add(Conv2D(50, kernel_size= kernel_size, padding='same'))
        model.add(Activation(activation=activ))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        #Second Convolutional layer:
        # model.add(Conv2D(50, kernel_size= kernel_size, padding='same'))
        # model.add(Activation(activation=activ))
        # model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        #Flatten layer:
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation(activation=activ))

        #Output layer:
        model.add(Dense(classes))
        model.add(Activation(activation='softmax'))

        return model

#Modle_phase_one.build_model(96,96,1,2)