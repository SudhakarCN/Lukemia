import cv2
import pandas as pd
import numpy as np
from keras.preprocessing import image


def generate_input(dataset):
    inputs = []
    data = dataset
    for loc in range(len(data.Location)):
        print("Processing for the input: ", loc)
        image_size = 224
        img = data.Location[loc]
        train_img = cv2.imread(img)
        resized_image = cv2.resize(train_img, (image_size, image_size))
        grayed_img = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        train_image = image.img_to_array(grayed_img) / 255
        train_image = train_image.astype("float32")
        # print(train_img)
        # print(train_image.shape)
        inputs.append(train_image)
    X = np.array(inputs)
    return X


def generate_output(dataset):
    train_output = []
    for loc in range(len(dataset.Label)):
        print("Processing output number ", loc)
        train_output.append(dataset.Label[loc])
    return train_output


def generating_train_data(dataset):
    train_dataset = dataset
    train_input = generate_input(train_dataset)
    print("Processed the inputs")
    train_output = generate_output(train_dataset)
    print("Processed the outputs")
    return train_input, train_output

def generating_test_data(dataset):
    test_dataset = dataset
    test_input = generate_input(test_dataset)
    print("Test inputs are processed")
    return test_input

