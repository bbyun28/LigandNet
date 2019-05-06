# Python script to train XGBClassifier
# Author: Md Mahmudulla Hassan
# Last modified: 05/05/2019

import pandas as pd
import os
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import  make_scorer, roc_auc_score, recall_score, accuracy_score, precision_score
from sklearn import metrics
import xgboost as xgb
import multiprocessing as mp
import pickle
import json
import os

# List the directories and files
active_dir = "actives_fingerprints"
decoy_dir = "decoys_fingerprints"
model_dir = 'xgb_models'
output_dir = 'outputs'

active_files = glob(os.path.join(active_dir, '*.csv'))
decoy_files = glob(os.path.join(decoy_dir, 'all', '*.csv'))
print("Number active and decoy files: {}, {}".format(len(active_files), len(decoy_files)))

def xgb_model(train_data, train_label, test_data, test_label, protein):
    # Define the filenames
    model_file = os.path.join(model_dir, protein + '.xgb')
    result_file = os.path.join(output_dir, "{}_results.json".format(protein))
    plot_file_logloss = os.path.join(output_dir, "{}_logloss.png".format(protein))
    plot_file_error = os.path.join(output_dir, "{}_classification_error.png".format(protein))

    
    actives_count = np.sum(train_label == 1).item() # Type casting from int32 to int for json serializable
    # Ignore the proteins that have less than 20 actives
    if actives_count < 20: 
        print("\tNUMBER OF ACTIVES ({}) IS NOT SUFFICIENT. ABORTING THE TRAINING".format(actives_count))
        return
    decoys_count = np.sum(train_label == 0).item() 
    ratio = float(decoys_count) / actives_count
    clf = xgb.XGBClassifier(max_depth=7,
                           n_jobs=mp.cpu_count(),
                           min_child_weight=1,
                           learning_rate=0.5,
                           n_estimators=1000,
                           silent=True,
                           objective='binary:logistic',
                           gamma=0,
                           max_delta_step=0,
                           subsample=1,
                           colsample_bytree=1,
                           colsample_bylevel=1,
                           reg_alpha=0,
                           reg_lambda=0,
                           scale_pos_weight=ratio,
                           seed=1,
                           missing=None)
    
    clf.fit(train_data, train_label, 
            eval_metric=['error', 'logloss'], 
            verbose=False,
            eval_set=[(train_data, train_label), (test_data, test_label)], 
            early_stopping_rounds=20)
    
    # Save the classifier
    with open(model_file, 'wb') as f:
        pickle.dump(clf, f)
    
    # Save the report
    report_file = os.path.join(output_dir, "{}_report.json".format(protein))
    y_pre = clf.predict(test_data)
    y_pro = clf.predict_proba(test_data)[:, 1]
    print("\tAUC Score : {}".format(metrics.roc_auc_score(test_label, y_pro)))
    print("\tAccuracy : {}".format(metrics.accuracy_score(test_label, y_pre)))
    print("\tF1 Score : {}".format(metrics.f1_score(test_label, y_pre)))
    print("\tREPORT: {}".format(metrics.classification_report(test_label, y_pre)))
    
    
    # Save the results
    results = dict()
    evals_result = clf.evals_result()
    results['protein_name'] = protein
    results['evals_result'] = {'train': evals_result['validation_0'], 'valid': evals_result['validation_1']}
    results['roc_auc'] = metrics.roc_auc_score(test_label, y_pro)
    results['accuracy'] = metrics.accuracy_score(test_label, y_pre)
    results['f1_score'] = metrics.f1_score(test_label, y_pre)
    results['data_info'] = {"train_count": len(train_data), 
                            "test_count": len(test_data),
                            "actives_count": actives_count, 
                            "decoys_count": decoys_count}
    
    with open(result_file, 'w') as f:
        json.dump(results, f)
    
    # Save the plots
    epochs = len(evals_result['validation_0']['error'])
    x_axis = range(0, epochs)
    
    # plot log loss
    fig = plt.figure(figsize=(10, 5))
    plt.plot(x_axis, evals_result['validation_0']['logloss'], label='Train')
    plt.plot(x_axis, evals_result['validation_1']['logloss'], label='Test')
    plt.legend()
    plt.ylabel('Logloss')
    plt.title('XGBoost Logloss ({})'.format(protein))
    plt.savefig(plot_file_logloss, format='png', dpi=1000)

    # plot classification error
    fig = plt.figure(figsize=(10, 5))
    plt.plot(x_axis, evals_result['validation_0']['error'], label='Train')
    plt.plot(x_axis, evals_result['validation_1']['error'], label='Test') # This is actually the validation error
    plt.legend()
    plt.ylabel('Classification Error')
    plt.title('XGBoost Classification Error ({})'.format(protein))
    plt.savefig(plot_file_error, format='png', dpi=1000)
    
    return True

def main():
    error_proteins = []
    for _file in active_files:
        _, protein = os.path.split(_file)
        protein, _ = os.path.splitext(protein)        
        model_file = os.path.join(model_dir, protein + '.xgb')        
        decoy_file = os.path.join(decoy_dir, 'all', "decoys_" + protein + ".csv")    
        if not os.path.isfile(decoy_file):
            print("\tERROR: DECOYS FOR {} NOT FOUND".format(protein))
            error_proteins.append(protein)
            continue
            
        print("TRAINING {}".format(protein))
        try:
            actives = pd.read_csv(_file, header=None)
            decoys = pd.read_csv(decoy_file, header=None)
        except Exception as e:
            print(str)
            error_proteins.append(protein)
            continue
            
        print("\tACTIVES: {}, DECOYS: {}".format(len(actives), len(decoys)))
        actives_x = actives.iloc[:, 1:].values
        actives_y = np.ones(len(actives_x))
        decoys_x = decoys.iloc[:, :].values
        decoys_y = np.zeros(len(decoys_x))
        x = np.concatenate((actives_x, decoys_x))
        y = np.concatenate((actives_y, decoys_y)) #labels

        # Split the data into train and test
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)
        xgb_model(x_train, y_train, x_test, y_test, protein)


if __name__=="__main__": main()