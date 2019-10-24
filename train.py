
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
sys.path.append(os.path.abspath('./src/networks'))

#To handel OOM errors
import tensorflow as tf
from keras import backend as K
import keras.backend.tensorflow_backend as ktf
def get_session():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction= 0.9,
                                allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
ktf.set_session(get_session())

#Standard imports
import pandas as pd
import numpy as np
from keras.optimizers import Adam, RMSprop, Nadam, SGD

#Custom imports
import config
from src.training import data_loader
from src.training.metrics import bce_dice_loss, dice_coeff
from src.training.seg_data_generator import SegDataGenerator
from src.training.keras_callbacks import get_callbacks
from src.training.training_modes import training_scratch, training_checkpoint, transfer_learning
from src.training.keras_history import generate_stats
from src.training.plots import save_plots

if __name__ == "__main__":

    dataset_path = config.dataset_path
    exp_name = config.exp_name

    train, val, test = data_loader.get_samples(dataset_path)

    print("\nPreparing dataset for Training")
    X_train, y_train = data_loader.build_source(train, dataset_path)

    print("\nPreparing dataset for Validation")
    X_val, y_val = data_loader.build_source(val, dataset_path)

    #Params
    tile_size = config.tile_size
    no_of_samples = config.no_of_samples
    downs = config.down_sampling

    batch_size = config.batch_size
    epochs = config.epochs
    initial_epoch = config.initial_epoch

    loss_class = {'bin_cross': 'binary_crossentropy',
                  'bce_dice': bce_dice_loss,
                 'wbce_dice': wbce_dice_loss}

    metric_class = {'dice':dice_coeff}

    optimiser_class = {'adam': (Adam, {}),
                       'nadam': (Nadam, {}),
                       'rmsprop': (RMSprop, {}),
                       'sgd':(SGD, {'decay':1e-6, 'momentum':0.90, 'nesterov':True})}

    training_frm_scratch = config.training_frm_scratch
    training_frm_chkpt = config.training_frm_chkpt
    transfer_lr = config.transfer_lr

    if sum((training_frm_scratch, training_frm_chkpt, fine_tuning, transfer_lr)) != 1:
        raise Exception("Conflicting training modes")

    #spe = Steps per epoch
    train_spe = int(np.floor((len(X_train)*no_of_samples*2) / batch_size)) # factor of 2 bcos of Augmentation
    val_spe = int(np.floor((len(X_val)*no_of_samples) / batch_size))


    # Initialise generators
    train_generator = SegDataGenerator(dataset_path, img_source=X_train,
                                    mask_source=y_train, batch_size= batch_size,
                                    no_of_samples = no_of_samples, tile_size= tile_size,
                                    downsampling_factor = downs, aug=True)

    val_generator = SegDataGenerator(dataset_path, img_source=X_val,
                                    mask_source=y_val, batch_size= batch_size,
                                    no_of_samples = no_of_samples, tile_size= tile_size,
                                    downsampling_factor = downs, aug=False)

    if training_frm_scratch:
        model, gpu_model = training_scratch(optimiser_class, loss_class, metric_class)

    elif training_frm_chkpt:
        model, gpu_model = training_checkpoint()

    elif fine_tuning:
        model, gpu_model = fine_tune(optimiser_class, loss_class, metric_class)

    elif transfer_lr:
        model, gpu_model = transfer_learning(optimiser_class, loss_class, metric_class)

    #Print the model params
    print("Model training params:")
    trainable_count = int(np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
    non_trainable_count = int(np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))
    params = (trainable_count + non_trainable_count,trainable_count, non_trainable_count)

    print('Total params: {:,}'.format(params[0]))
    print('Trainable params: {:,}'.format(params[1]))
    print('Non-trainable params: {:,}'.format(params[2]))

    #Set callbacks
    callbacks_list = get_callbacks(model)

    # Start/resume training
    if config.no_of_gpu > 1:
        history = gpu_model.fit_generator(steps_per_epoch= train_spe,
                                          generator=train_generator,
                                          epochs=epochs,
                                          validation_data = val_generator,
                                          validation_steps = val_spe,
                                          initial_epoch = initial_epoch,
                                          callbacks = callbacks_list)

    else:
        history = model.fit_generator(steps_per_epoch= train_spe,
                                  generator=train_generator,
                                  epochs=epochs,
                                  validation_data = val_generator,
                                  validation_steps = val_spe,
                                  initial_epoch = initial_epoch,
                                  callbacks = callbacks_list)

    #Save final complete model
    filename = "model_ep_"+str(int(epochs))+"_batch_"+str(int(batch_size))
    model.save("./data/"+exp_name+"/"+filename+".h5")
    print("Saved complete model file at: ", filename+"_model"+".h5")

    #Save history
    history_to_save = generate_stats(history, config)
    pd.DataFrame(history_to_save).to_csv("./data/"+exp_name+"/"+filename + "_train_results.csv")
    save_plots(history, exp_name)
