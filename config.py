
dataset_path = "data/datasets/AerialImageDataset/train"
exp_name = "inria_linknet"

tile_size = 512
down_sampling = 2.0
no_of_gpu = 1
batch_size = 12*no_of_gpu #has to be even no.

no_of_samples = 700 #no_of_samples_perunit for inria
perce_aoi_touse = 0.1 #decides the no of aois to use for training and validation.
#class_weight = {0: 1.0, 1: 1.0}  #0 :Background, 1 : Building

epochs = 30
patience_lr = 5
factor_lr = 0.75
min_delta = 0.01
patience_es = 10

dice_weight=0.5
cross_entropy_weight=0.5

training_frm_scratch = True
training_frm_chkpt = False
transfer_lr = False
trial = False

if training_frm_scratch:
    model = 'linknet' #used for importing from src networks
    initial_epoch = 0
    optimiser = 'sgd'  #enter everything in small letters
    loss = 'bce_dice'
    metric = 'dice'
    learning_rate = 0.001

if training_frm_chkpt:
    initial_epoch = 10 #training starts from 'initial epoch + 1'
    model_path = "path/to/checkpoint-10-0.90.h5"

if fine_tuning:
   model = "newmodel"
   weights_path="data/pretrained/newmodel/checkpoint-newmodel.h5"

   initial_epoch = 0

   optimiser = 'sgd'  #enter everything in small letters
   loss = 'wbce_dice'
   metric = 'dice'
   learning_rate = 0.0001

if transfer_lr:
    model_path = "path/to/checkpoint-10-0.90.h5"
    model = "linknet"
    initial_epoch = 0

    trainable_layers =  list(range(104)) #complete architecture

    optimiser = 'sgd'  #enter everything in small letters
    loss = 'bce_dice'
    metric = 'dice'
    learning_rate = 0.0001

if trial:
    print("Trial Mode Activated")
    epochs = 3
    no_of_samples = 10
    perce_aoi_touse = 0.10
