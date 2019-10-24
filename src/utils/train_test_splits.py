import os
import glob
import numpy as np

base = "/Users/pradip.gupta/jiogis/data/datasets/AerialImageDataset/train/"
config = {"train":0.80,
          "validation":0.19,
          "test":0.01}

            
def create_splits(base, config):
    
    phases = ["train", "test", "validation"]
    
    if os.path.exists(base + "train.txt"):
        os.remove(base + "train.txt")
    
    if os.path.exists(base + "validation.txt"):
        os.remove(base + "validation.txt")
        
    if os.path.exists(base + "test.txt"):
        os.remove(base + "test.txt")
    
    images_list = glob.glob(base + "images/" + "**/*.tif", recursive=True)
    aois = [os.path.basename(fname)[:-4] for fname in images_list]
    nos = len(aois)
    
    np.random.shuffle(aois)
    
    train_size = int(np.ceil(nos*config["train"]))
    val_size = int(np.ceil(nos*config["validation"]))
    
    images = {}
    images["train"] = aois[:train_size]
    images["validation"] = aois[train_size:(train_size+val_size)]
    images["test"] = aois[(train_size+val_size):]
    
    
    for phase in phases:
        with open('{}{}.txt'.format(base,phase), 'a') as f:
            for image in images[phase]:
                f.write('{}\n'.format(image))


if __name__ == "__main__":
    
    create_splits(base, config)
    
        
    
    
    
    
    
    




