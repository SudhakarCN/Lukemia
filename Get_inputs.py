import os
import pandas as pd

#-----Creating a dataset for training purpose:
Cancer_train = pd.DataFrame(columns=["Image_Name", "Cell_Count", "Label", "Location"])
folders = ["fold_0","fold_1","fold_2"] #------->Folder which contains the training images

for folder in folders:
    path = "/home/sudhakar/Documents/Dataset/VVHN/C-NMC_Leukemia/C-NMC_training_data/" + folder
    for roots,dirs,files in os.walk(path):
        for file in files:
            if file == ".DS_Store":
                continue
        else:
                cell_count = [file.split("_")][0][3]
                if "all" in file:
                    label = 1
                elif "hem" in file:x
                    label = 0
            Cancer_train = Cancer_train.append({'Image_Name':file,"Cell_Count":cell_count,'Label':label, 'Location': roots + "/" + file},ignore_index=True)

print(Cancer_train.head())
print(Cancer_train.tail())
Cancer_train.to_csv("Training_data.csv")
