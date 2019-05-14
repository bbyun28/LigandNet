from __future__ import print_function
import pandas as pd
import numpy as np
import sys
sys.path.append("../ddt/")
from utility import FeatureGenerator
from sklearn.externals import joblib
import os
import multiprocessing as mp
import json
import warnings
import time
warnings.filterwarnings("ignore")


MODELS_DIR = "models"
PY_VERSION = '27' if sys.version_info[0] < 3 else '35'


def get_models():    
    model_names = open('py' + PY_VERSION + '_models.txt', 'r').readlines()[0].split(',')
    for model in model_names:
        yield model, joblib.load(os.path.join(MODELS_DIR, model))

def get_prediction(features, confidence=0.9):
    actives = []
    for model_name, model in get_models():
        print('predicting using {}'.format(model_name))
        if type(model).__name__ == "SVC":
            pred = model.predict(features)
            if pred[0] == 1.0: actives.append((model_name,))
        else:
            pred = model.predict_proba(features)
            model_conf = pred[0][1]
            if model_conf > confidence: actives.append((model_name, str(model_conf)))
    return actives
    
#TODO: extend functionality for sdf files. there might be multiple compounds in sdfs. provide compound specific results

if __name__=="__main__":
    start = time.time()
    import argparse
    parser = argparse.ArgumentParser(description="Prediction using LigandNet models")
    parser.add_argument('--smiles', action='store', dest='smiles', required=False, help='SMILES string')
    parser.add_argument('--sdf', action='store', dest='sdf', required=False, help='SDF file location')
    parser.add_argument('--out', action='store', dest='out', required=False, help='Output file')
    args = parser.parse_args()
    
    # At least one of the inputs is required
    if not (args.smiles or args.sdf):
        parser.error('No input is given, add --smiles or --sdf')
    
    parse_dict = vars(args)
    
    if args.sdf and not os.path.isfile(parse_dict['sdf']):
        print("ERROR: SDF file {} not found".format(parse_dict['sdf']))
        sys.exit(1)
    
    # Extract features
    ft = FeatureGenerator()
    if args.smiles:
        ft.load_smiles(str(parse_dict['smiles']))
    elif args.sdf:
        ft.load_sdf(parse_dict['sdf'])
    
    features = ft.extract_tpatf()
    features = np.array(features).reshape((-1, 2692))
    results = get_prediction(features)
    
    if len(results) < 1: 
        print("No active protein found")
        sys.exit(1)
        
    output = {'smiles': parse_dict['smiles'], 'active_proteins': results}
    
    # Write the output
    output_dir = parse_dict['out'] if args.out else 'output'
    if not os.path.isdir(output_dir): os.makedirs(output_dir)
    with open(os.path.join(output_dir, 'output.json'), 'w') as f:
        json.dump(output, f)  
    
    end = time.time()
    print("Execution time: {} seconds.".format(round(end - start, 2)))
    sys.exit(1)

