import os
import pandas as pd
from src.utils.buzz_utility import raster_explorer
import config

def get_samples(dataset_path):
    
    train = pd.read_csv(os.path.join(dataset_path, "train.txt"), header=None)
    val = pd.read_csv(os.path.join(dataset_path, "validation.txt"), header=None)
    test = pd.read_csv(os.path.join(dataset_path, "test.txt"), header=None)
    
    if not config.perce_aoi_touse == 1.0:
        train = train[:int(len(train)*config.perce_aoi_touse)]
        val = val[:int(len(val)*config.perce_aoi_touse)]
    
    print("No of aois:")
    print("for train:", len(train))
    print("for validation:", len(val))
    print("for test:", len(test))
    
    return train, val, test


def build_source(df, dataset_path):
    
    X = []
    y = []
    
    for file in df[0].values:
        if not file.startswith("."):            
            rgb_path = dataset_path + "/images/" + file+".tif"
            binary_path = dataset_path + "/masks/" + file+".tif" 
            
            print("\nAdding {} to AOI list".format(file))

            ds_rgb = raster_explorer(rgb_path, stats=False)
            ds_binary = raster_explorer(binary_path, stats=False)
            
            X.append(ds_rgb)
            y.append(ds_binary)
            
    return X, y