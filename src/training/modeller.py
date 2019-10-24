import config

def finetune_model(base_model):
        
    for i, layer in enumerate(base_model.layers):
        if i in config.trainable_layers:
            layer.trainable=True
        else:
            layer.trainable=False         
    
    print("Layer freezing complete!!")
    
    return base_model