import pandas as pd
from Phase_One import gnet
from Phase_One.Image_Processing import generating_train_data
from Phase_One.leNet_model import leNet
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import pickle
from keras.optimizers import SGD, Adam
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

# dataset_loc = input("Enter the location of the dataset: ")
train_dataset = pd.read_csv("Training1_data.csv")

#Plotting training data:
def show_train_history(history, xlabel, ylabel, train, name):
    for item in train:
        plt.plot(history[item])
    plt.title('Train History')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(train, loc='upper left')
    plt.savefig("")
    plt.show()

#Constants:
epochs = [20,30,20,30]
optms = ['Adam', 'SGD', 'Adam', 'SGD']
lr = 1e-3
batch_size = 16
epoch_steps = int(4323/batch_size)

X,y = generating_train_data(train_dataset)
# print(X)
# print(len(X))
# print(X.shape)
# print(y)
# print(len(y))

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=4)
# print(X_train)
# print(X_test.shape)
# print(y_test)
# print(len(y_train))

#One hot encoding:
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
output_train = y_train, y_train, y_train
output_test = y_test,y_test,y_test
# print(X_train.shape)
# print(y_train.shape)

# aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")
# generator_main = aug.flow(x=X_train, y=y_train ,batch_size=batch_size)
#
# def my_generator(generator):
#     while True:
#         data = next(generator)
#         x = data[0]
#         y = data[1], data[1], data[1]
#         yield x, y
#
# train_generator = my_generator(generator_main)

# #Fit the model:

print("<-----------------------------Compiling Model----------------------------->")
model = gnet.googleNet(width=X.shape[1], height=X.shape[2], depth=X.shape[3], classes=2)
model.summary()
for i in range(len(optms)):
    print("GoogleNet Compilation: {0} | Optimizer: {1} | Epochs: {2}".format(i+1, optms[i], epochs[i]))
    model.compile(optimizer=optms[i], loss="categorical_crossentropy", metrics=["accuracy"])
    #History = model.fit_generator(train_generator, steps_per_epoch=epoch_steps, epochs=epochs[i], shuffle=True)
    History = model.fit(x=X_train, y= output_train, batch_size=batch_size, epochs=epochs, verbose=1)
    show_train_history(History, 'Epoch', 'Accuracy', ('main_Acc', 'aux1_acc', 'aux2_acc'), name="GoogleNet Compilation{0}".format(i))

#Evaluation:
score = model.evaluate(x=X_test, y=y_test, verbose=1)
print("The score is: ", round(score[1]*100,2))

pickle.dump(model, open('gnet', 'wb'))



# Training_data.csv

