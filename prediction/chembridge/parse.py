# Parser script for converting the **WEIRDLY** saved chembridge results into a readable format.
# Md Mahmudulla Hassan
# Last modified: 10/17/2018

import glob
import pandas as pd

files = glob.glob("*/*.txt", recursive=True)

predictions = []
for _file in files:
    model_name = _file.split("classifier_")[1][:-4]
    with open(_file, 'r') as f:
        lines = f.readlines()
        if len(lines) == 0: continue
        lines = [l.split('\n')[0] for l in lines]
        lines = [l.split(' ') for l in lines]
        if len(lines[0]) != 1:
            # with probabilities
            result = [[model_name, l[0].split('\t')[0], l[2].split(']')[0]] for l in lines]
            predictions.extend(result)
        else:
            # SVM models. No probabilities
            result = [[model_name, l[0].split('\t')[0], 'svm'] for l in lines]
            predictions.extend(result)

# create dataframe
df = pd.DataFrame.from_records(predictions, columns=['protein', 'id', 'probability'], index=None)
# save it
df.to_csv("predictions.csv", index=None)
# count the number of hits found for each of the proteins
counts = df.groupby('protein')['id'].count()
counts.reset_index().to_csv("counts.csv", index=None, header=["protein", "count"])
