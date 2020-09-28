import pandas as pd
import os
from Phase_One import Image_Processing
import pickle


pd.set_option('display.max_columns', 500)
test_dataset_location = input("Enter the location of the test dataset: ")

test_dataset = pd.read_csv(test_dataset_location)
#print(test_dataset.head())

predicted_dataset = pd.DataFrame(columns=["Patient_ID", "New_names", "Location", "Actual_label", "Predicted_label"])
predicted_dataset["Patient_ID"] = test_dataset["Patient_ID"]
predicted_dataset["New_names"] = test_dataset["new_names"]
predicted_dataset["Actual_label"] = test_dataset["labels"]
#print(predicted_dataset.head())

path = "/home/sudhakar/Documents/Dataset/VVHN/C-NMC_Leukemia/C-NMC_test_prelim_phase_data/C-NMC_test_prelim_phase_data"
loc = []
for root, dir, files in os.walk(path):
    for file in files:
        location = root + "/" + file
        loc.append(location)
    print("Completed getting the location of the test data")

predicted_dataset["Location"] = loc
#print(predicted_dataset.head())

test_inputs = Image_Processing.generate_input(predicted_dataset)

model = pickle.load(open("Leukemia_phase_one.pkl", "rb"))
predicted_label = model.predict(test_inputs)
print(predicted_label)


#/home/sudhakar/Documents/Dataset/VVHN/C-NMC_Leukemia/C-NMC_test_prelim_phase_data/C-NMC_test_prelim_phase_data_labels.csv