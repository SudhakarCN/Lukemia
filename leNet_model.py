from keras.models import Sequential
from keras.layers.convolutional import Conv2D, AveragePooling2D
from keras.layers.core import Activation, Dense, Flatten

class leNet():
    @staticmethod
    def build_model(width, height, depth, classes):
        #Initialize he model
        model = Sequential()
        input_shape = (height,width, depth)
        activ = 'tanh'
        kernel_size = (5,5)

        #Convolution layer 1:
        model.add(layer=Conv2D(6, kernel_size=kernel_size, strides=(1,1), input_shape=input_shape, activation=activ, padding='same'))

        #Pooling layer 1:
        model.add(AveragePooling2D(pool_size=(2,2), strides=(1,1), padding='valid'))

        #Convolution layer 2:
        model.add(layer=Conv2D(16, kernel_size=kernel_size, strides=(1,1), activation=activ, padding='valid'))

        #Pooling layer 2:
        model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
        

        #Fully connected convolutional layer:
        model.add(layer=Conv2D(120, kernel_size=kernel_size, strides=(1,1), padding='valid', activation=activ))

        #Flatten:
        model.add(layer=Flatten())

        #Fully Connected layer:
        model.add(layer=Dense(84, activation=activ))

        #Output softmax layer:
        model.add(layer=Dense(classes, activation='softmax'))

        return model
        