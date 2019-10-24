
import os, sys
from unipath import Path

absolute_path = Path('src/networks/').absolute()
sys.path.append(str(absolute_path))

import importlib
import json
from keras.models import model_from_json, load_model

class LoadModel:
    """
    model_dict = {"weights_file":"",
                "arch_file":"",
                "model_file":""}

    """
    @classmethod
    def load(cls, **kwargs):
        
        if "model_file" in kwargs:
            model = cls.load1(kwargs.get("model_file"))
            
            return model
        
        elif "model_name" in kwargs:
            model = cls.load3(kwargs["model_name"],kwargs["weight_file"])
            
            return model
        
        if kwargs["weight_file"] and kwargs["json_file"]:
            model = cls.load2(kwargs["weight_file"], kwargs["json_file"])
            
            return model

    @staticmethod
    def load1(cls, model_path):
        model = load_model(model_path)
        
        return model
        
    @classmethod
    def load2(cls, weight_file, json_file):
        
        jsonfile = open(json_file,'r')
        loaded_model_json = jsonfile.read()
        jsonfile.close()
        
        model = model_from_json(loaded_model_json)
        model.load_weights(weight_file)
        
        return model
    
    @classmethod    
    def load3(cls, model_name, weights_path):
        
        build = getattr(importlib.import_module("higal_unet_resnet50"),"build")
        model = build(256, 3)
        model.load_weights(weights_path)
    
    @classmethod    
    def load4(cls, custom):        
#       kwargs.custom   = {'bce_dice_loss':bce_dice_loss,'dice_coeff':dice_coeff}
        model = load_model("data/pretrained/savera/SAVERA_86_trained_weights.best.hdf5.index", 
                           custom_objects=custom)
        
        return model
        