def generate_stats(history, config):

    history_to_save = history.history
    
    history_to_save['train acc'] = max(history.history['acc'])
    history_to_save['train dice_coeff'] = max(history.history['dice_coeff'])
    history_to_save['train loss'] = min(history.history['loss'])
    history_to_save['last train accuracy'] = history.history['dice_coeff'][-1]
    history_to_save['last train loss'] = history.history['loss'][-1]
    
    history_to_save['val acc'] = max(history.history['val_acc'])
    history_to_save['val dice_coeff'] = max(history.history['val_dice_coeff'])
    history_to_save['val loss'] = min(history.history['val_loss'])
    history_to_save['last val loss'] = history.history['val_loss'][-1]
    history_to_save['last val acc'] = history.history['val_dice_coeff'][-1]
    
    history_to_save['final lr'] = history.history['lr'][-1]
    history_to_save['total epochs'] = len(history.history['lr'])
    
    
    history_to_save['downsampling_factor'] = config.down_sampling
    history_to_save['initial lr'] = config.learning_rate
    history_to_save['optimiser'] = config.optimiser 
    history_to_save['loss'] = config.loss
    history_to_save['metric'] = config.metric
    history_to_save['dice_weight'] = config.dice_weight
    history_to_save['cross_entropy_weight'] = config.cross_entropy_weight


    return history_to_save

