import tensorflow as tf
import os

def get_var_to_restore_list(ckpt_path, mask=[], prefix="", ignore_list=[]):
    """
    Get all the variable defined in a ckpt file and add them to the returned var_to_restore list. Allows for partially defined model to be restored fomr ckpt files.
    Args:
        ckpt_path: path to the ckpt model to be restored
        mask: list of layers to skip
        prefix: prefix string before the actual layer name in the graph definition
    """
    variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    variables_dict = {}
    for v in variables:
        name = v.name[:-2]
        #print(name)
        skip=False
        #check for skip
        for m in mask:
            if m in name:
                skip=True
                continue
        if not skip:
            variables_dict[v.name[:-2]] = v

    #print('====================================================')
    reader = tf.train.NewCheckpointReader(ckpt_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    var_to_restore = {}
    for key in var_to_shape_map:
        t_key=key
        #print(key)
        for ig in ignore_list:
            t_key=t_key.replace(ig,'')
        if prefix+t_key in variables_dict.keys():
            var_to_restore[key] = variables_dict[prefix+t_key]
    
    return var_to_restore


def check_for_weights_or_restore_them(logdir, session, initial_weights=None, prefix='', ignore_list=[]):
    """
    Check for the existance of a previous checkpoint in logdir, if not found and weights is set to a valid path restore that model instead.
    Args:
        log_dir: dir where to look for previous checkpoints
        session: tensorflow session to restore weights
        initial_weights: optional fall back weights to be used if no available weight as been found
        prefix: prefix to be putted before variable names in the ckpt file
    Returns:
        A boolean that states if the weights have been restored or not and the number of step restored (if any)
    """
    ckpt = tf.train.latest_checkpoint(logdir)
    if ckpt:
        print('Found valid checkpoint file: {}'.format(ckpt))
        var_to_restore = get_var_to_restore_list(ckpt, [], prefix="")
        restorer = tf.train.Saver(var_list=var_to_restore)
        restorer.restore(session,ckpt)
        step = int(ckpt.split('-')[-1])
        return True,step
    elif initial_weights is not None:
        if os.path.isdir(initial_weights):
            #if its a directory fetch the last checkpoint
            initial_weights = tf.train.latest_checkpoint(initial_weights)
        step = 0
        var_to_restore = get_var_to_restore_list(initial_weights, [], prefix=prefix, ignore_list=ignore_list)
        print('Found {} variables to restore in {}'.format(len(var_to_restore),initial_weights))
        if len(var_to_restore)>0:
            restorer = tf.train.Saver(var_list=var_to_restore)
            restorer.restore(session, initial_weights)
            return True,0
        else:
            return False,0
    else:
        print('Unable to restore any weight')
        return False,0
