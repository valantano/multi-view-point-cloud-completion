import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from base.scaphoid_utils.logger import print_log


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def ensure_path_exists(path, logger=None):
    if not os.path.exists(path):
        print_log(f'[RESUME INFO] no checkpoint file from path {path}...', logger = logger, color='red')
        raise NotImplementedError('no checkpoint file from path %s...' % path)
    
    
def load_ckpt_into_model(base_model, base_ckpt, freeze, logger=None):
    model_keys = set(base_model.state_dict().keys())
    ckpt_keys = set(base_ckpt.keys())
    missing_keys = model_keys - ckpt_keys
    unexpected_keys = ckpt_keys - model_keys

    # if missing_keys:
    #     print_log(f"[WARNING] Missing keys in checkpoint: {missing_keys}", logger=logger)
    if unexpected_keys:
        print_log(f"[WARNING] Unexpected keys in checkpoint: {sorted(unexpected_keys)}", logger=logger, color='red')
        print_log(f"Required keys in model: {sorted(model_keys)}", logger=logger, color='yellow')
        raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected_keys}")
    else:
        print_log(f"[INFO] No unexpected keys in checkpoint.", logger=logger, color='green')

    successfully_loaded_keys = []
    base_model.load_state_dict(base_ckpt, strict=False)
    successfully_loaded_keys = base_ckpt.keys()

    if freeze:
        for name, param in base_model.named_parameters():
            if name in successfully_loaded_keys:
                param.requires_grad = False
                print_log(f'[FREEZE INFO] Layer {name} has been frozen.', logger=logger)


def init_rot_extractors(base_model, args, freeze=True, logger = None):
    ckpt_path_dorsal = Path(os.path.join(args.experiment_path, '../../../RotationCfgs/ScaphoidPointAttN_rotation_RERG_dorsal/+RERGDorsal', 'ckpt-best.pth')).resolve()
    ckpt_path_volar = Path(os.path.join(args.experiment_path, '../../../RotationCfgs/ScaphoidPointAttN_rotation_RERG_volar/+RERGVolar', 'ckpt-best.pth')).resolve()
    print_log(f"[RESUME INFO] Loading dorsal and volar RERGs...", logger = logger, color='yellow')

    ensure_path_exists(ckpt_path_dorsal, logger=logger)
    ensure_path_exists(ckpt_path_volar, logger=logger)

    print_log(f'[RESUME INFO] Loading model weights from {ckpt_path_volar} and {ckpt_path_dorsal}...', logger)

    # load state dict
    map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
    state_dict = torch.load(ckpt_path_dorsal, map_location=map_location, weights_only=False)

    base_ckpt_fe_d = {k.replace(".0.", ".1.RE."): v for k, v in state_dict['base_model'].items() if '.0.' in k}
    base_ckpt_sg_d = {k.replace(".1.", ".1.RG."): v for k, v in state_dict['base_model'].items() if '.1.' in k}

    state_dict = torch.load(ckpt_path_volar, map_location=map_location, weights_only=False)
    base_ckpt_fe_v = {k.replace(".0.", ".0.RE."): v for k, v in state_dict['base_model'].items() if '.0.' in k}
    base_ckpt_sg_v = {k.replace(".1.", ".0.RG."): v for k, v in state_dict['base_model'].items() if '.1.' in k}

    base_ckpt = {**base_ckpt_fe_d, **base_ckpt_sg_d, **base_ckpt_fe_v, **base_ckpt_sg_v}

    load_ckpt_into_model(base_model, base_ckpt, freeze, logger=logger)


def init_pose_estimators(base_model, args, freeze=True, logger = None):
    ckpt_path_dorsal = Path(os.path.join(args.experiment_path, '../../../PoseEstCfgs/PoseEst_ScaphoidPointAttN_dorsal/FPoseDorsalBest0', 'ckpt-best.pth')).resolve()
    ckpt_path_volar = Path(os.path.join(args.experiment_path, '../../../PoseEstCfgs/PoseEst_ScaphoidPointAttN_volar/FPoseVolarBest0', 'ckpt-best.pth')).resolve()

    # ckpt_path_dorsal = Path(os.path.join(args.experiment_path, '../../../PoseEstCfgs/PoseEst_ScaphoidPointAttN_dorsal/1PoseDLossWeight', 'ckpt-last.pth')).resolve()
    # ckpt_path_volar = Path(os.path.join(args.experiment_path, '../../../PoseEstCfgs/PoseEst_ScaphoidPointAttN_volar/1PoseVLossWeight', 'ckpt-last.pth')).resolve()
    print_log(f"[RESUME INFO] Loading dorsal and volar Pose Estimators...", logger = logger, color='yellow')

    ensure_path_exists(ckpt_path_dorsal, logger=logger)
    ensure_path_exists(ckpt_path_volar, logger=logger)

    print_log(f'[RESUME INFO] Loading model weights from {ckpt_path_volar} and {ckpt_path_dorsal}...', logger = logger )

    # load state dict
    map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
    state_dict = torch.load(ckpt_path_dorsal, map_location=map_location, weights_only=False)

    base_ckpt_fe_d = {k.replace(".0.", ".2."): v for k, v in state_dict['base_model'].items() if '.0.' in k}
    base_ckpt_sg_d = {k.replace(".1.", ".3."): v for k, v in state_dict['base_model'].items() if '.1.' in k}

    state_dict = torch.load(ckpt_path_volar, map_location=map_location, weights_only=False)
    base_ckpt_fe_v = {k.replace(".0.", ".0."): v for k, v in state_dict['base_model'].items() if '.0.' in k}
    base_ckpt_sg_v = {k.replace(".1.", ".1."): v for k, v in state_dict['base_model'].items() if '.1.' in k}

    base_ckpt = {**base_ckpt_fe_d, **base_ckpt_sg_d, **base_ckpt_fe_v, **base_ckpt_sg_v}

    load_ckpt_into_model(base_model, base_ckpt, freeze, logger=logger)


def partly_resume_model_pose_both(base_model, args, config, freeze=True, logger = None):
     # ckpt_path = Path(os.path.join(args.experiment_path, '../../../BaselineCfgs/ScaphoidPointAttN_Baseline_Min_dorsal/+BaseMinDW', 'ckpt-last.pth')).resolve()

    ckpt_path = Path(os.path.join(args.experiment_path, '../../../StaticCfgs/ScaphoidPointAttN_Baseline_Max_static/X600StaticBaseMax0', 'ckpt-last.pth')).resolve()
    # ckpt_path = Path(os.path.join(args.experiment_path, '../../../StaticCfgs/ScaphoidPointAttN_Baseline_Max_static/+BaseMaxStatic600', 'ckpt-last.pth')).resolve()

    print_log(f"[RESUME INFO] Loading dorsal static model...", logger = logger, color='yellow')

    print_log(f"[RESUME INFO] Loading base model from {ckpt_path}...", logger = logger, color='yellow')

    if not os.path.exists(ckpt_path):
        print_log(f'[RESUME INFO] no checkpoint file from path {ckpt_path}...', logger = logger, color='red')
        raise NotImplementedError('no checkpoint file from path %s...' % ckpt_path)
    
    print_log(f'[RESUME INFO] Loading model weights from {ckpt_path}...', logger = logger )

    # load state dict
    map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
    state_dict = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    # parameter resume of base model
    # if args.local_rank == 0:
    base_ckpt = {k: v for k, v in state_dict['base_model'].items()}
    module_base_ckpt = {k.replace("module.", ""): v for k, v in base_ckpt.items()}

    base_ckpt_fe = {k.replace(".0.", ".0."): v for k, v in state_dict['base_model'].items() if '.0.' in k}
    base_ckpt_sg = {k.replace(".1.", ".1."): v for k, v in state_dict['base_model'].items() if '.1.' in k}
    base_ckpt = {**base_ckpt_fe, **base_ckpt_sg}

    base_ckpt = {k.replace("module.net_blocks.0.", "module.net_blocks.0.pose_extractor."): v for k, v in base_ckpt.items() }
    base_ckpt = {k.replace("module.net_blocks.1.", "module.net_blocks.0.pose_extractor."): v for k, v in base_ckpt.items() }

    load_ckpt_into_model(base_model, base_ckpt, freeze, logger=logger)


def partly_resume_model(base_model, args, config, freeze=True, logger = None):
    # ckpt_path = Path(os.path.join(args.experiment_path, '../../../BaselineCfgs/ScaphoidPointAttN_Baseline_Min_dorsal/+BaseMinDW', 'ckpt-last.pth')).resolve()
    if config.dataset.transform_with == 'dorsal':
        # ckpt_path = Path(os.path.join(args.experiment_path, '../../../StaticCfgs/ScaphoidPointAttN_Baseline_Min_dorsal_static/0DorsalStatic', 'ckpt-best.pth')).resolve()
        # ckpt_path = Path(os.path.join(args.experiment_path, '../../../StaticCfgs/ScaphoidPointAttN_Baseline_Min_dorsal_static/+BaseDorsalStatic', 'ckpt-last.pth')).resolve()
        ckpt_path = Path(os.path.join(args.experiment_path, '../../../StaticCfgs/ScaphoidPointAttN_Baseline_Min_dorsal_static/X600StaticDorsal0', 'ckpt-best.pth')).resolve()
        print_log(f"[RESUME INFO] Loading dorsal static model...", logger = logger, color='yellow')
    elif config.dataset.transform_with == 'volar':
        # ckpt_path = Path(os.path.join(args.experiment_path, '../../../StaticCfgs/ScaphoidPointAttN_Baseline_Min_volar_static/0VolarStatic', 'ckpt-best.pth')).resolve()
        # ckpt_path = Path(os.path.join(args.experiment_path, '../../../StaticCfgs/ScaphoidPointAttN_Baseline_Min_volar_static/+BaseVolarStatic', 'ckpt-last.pth')).resolve()
        ckpt_path = Path(os.path.join(args.experiment_path, '../../../StaticCfgs/ScaphoidPointAttN_Baseline_Min_volar_static/X600StaticVolarPre0', 'ckpt-best.pth')).resolve()
        print_log(f"[RESUME INFO] Loading volar static model...", logger = logger, color='yellow')
    elif config.dataset.transform_with == 'full':
        # ckpt_path = Path(os.path.join(args.experiment_path, '../../../StaticCfgs/ScaphoidPointAttN_Baseline_Min_dorsal_static/0DorsalStatic', 'ckpt-best.pth')).resolve()
        # ckpt_path = Path(os.path.join(args.experiment_path, '../../../StaticCfgs/ScaphoidPointAttN_Baseline_Max_static/+BaseMaxStatic/', 'ckpt-last.pth')).resolve()
        raise NotImplementedError(f"Depricated: Dataset transforms {config.dataset.transform_with} not supported for static model loading.")
        print_log(f"[RESUME INFO] Loading dorsal static model...", logger = logger, color='yellow')
    else:
        raise NotImplementedError(f"Dataset transforms {config.dataset.transform_with} not supported for static model \
                                  loading.")

    print_log(f"[RESUME INFO] Loading base model from {ckpt_path}...", logger = logger, color='yellow')

    if not os.path.exists(ckpt_path):
        print_log(f'[RESUME INFO] no checkpoint file from path {ckpt_path}...', logger = logger, color='red')
        raise NotImplementedError('no checkpoint file from path %s...' % ckpt_path)
    
    print_log(f'[RESUME INFO] Loading model weights from {ckpt_path}...', logger = logger )

    # load state dict
    map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
    state_dict = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    # parameter resume of base model
    # if args.local_rank == 0:
    base_ckpt = {k: v for k, v in state_dict['base_model'].items()}
    module_base_ckpt = {k.replace("module.", ""): v for k, v in base_ckpt.items()}

    base_ckpt_fe = {k.replace(".0.", ".0."): v for k, v in state_dict['base_model'].items() if '.0.' in k}
    base_ckpt_sg = {k.replace(".1.", ".1."): v for k, v in state_dict['base_model'].items() if '.1.' in k}
    base_ckpt = {**base_ckpt_fe, **base_ckpt_sg}

    load_ckpt_into_model(base_model, base_ckpt, freeze, logger=logger)


def ckpt_key_renamer_for_older_models(base_ckpt):
    """
    Ensure backwards compatibility for older checkpoints.
    """
    ckpt_renamed = {k.replace(".refine.conv_1.", ".refine.conv_shape1."): v for k, v in base_ckpt.items()}
    ckpt_renamed = {k.replace(".refine.conv_11.", ".refine.conv_shape."): v for k, v in ckpt_renamed.items()}

    ckpt_renamed = {k.replace(".refine.conv_x.", ".refine.conv_seeds."): v for k, v in ckpt_renamed.items()}
    ckpt_renamed = {k.replace(".refine.conv_x1.", ".refine.conv_seeds1."): v for k, v in ckpt_renamed.items()}

    ckpt_renamed = {k.replace(".refine1.conv_1.", ".refine1.conv_shape1."): v for k, v in ckpt_renamed.items()}
    ckpt_renamed = {k.replace(".refine1.conv_11.", ".refine1.conv_shape."): v for k, v in ckpt_renamed.items()}

    ckpt_renamed = {k.replace(".refine1.conv_x.", ".refine1.conv_seeds."): v for k, v in ckpt_renamed.items()}
    ckpt_renamed = {k.replace(".refine1.conv_x1.", ".refine1.conv_seeds1."): v for k, v in ckpt_renamed.items()}

    return ckpt_renamed


def resume_model(base_model, args, config, logger = None) -> tuple[int, float]:
    ckpt_path = os.path.join(args.experiment_path, 'ckpt-last.pth' if not args.test else 'ckpt-best.pth')
    if not os.path.exists(ckpt_path):
        print_log(f'[RESUME INFO] no checkpoint file from path {ckpt_path}...', logger = logger)
        return 0, 0
    print_log(f'[RESUME INFO] Loading model weights from {ckpt_path}...', logger = logger )

    map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
    state_dict = torch.load(ckpt_path, map_location=map_location, weights_only=False)

    base_ckpt = {k: v for k, v in state_dict['base_model'].items()}

    base_ckpt = ckpt_key_renamer_for_older_models(base_ckpt)
    
    if not list(base_model.state_dict().keys())[0].startswith('module.'):
        base_ckpt = {k.replace("module.", ""): v for k, v in base_ckpt.items()}
        
    base_model.load_state_dict(base_ckpt)


    # parameter
    start_epoch = state_dict['epoch'] + 1
    best_metrics = state_dict['best_metrics']
    if not isinstance(best_metrics, dict):
        best_metrics = best_metrics.state_dict()
    # print(best_metrics)
    best_cdl1_score = best_metrics.get('CDL1')

    print_log(f'[RESUME INFO] resume ckpts @ {start_epoch - 1} epoch( best_metrics = {str(best_metrics):s})', logger)
    return start_epoch, best_cdl1_score


def resume_optimizer(optimizer, args, logger = None):
    ckpt_path = os.path.join(args.experiment_path, 'ckpt-last.pth')
    if not os.path.exists(ckpt_path):
        print_log(f'------[RESUME INFO] no checkpoint file from path {ckpt_path}...', logger)
        return 0, 0, 0
    print_log(f'------[RESUME INFO] Loading optimizer from {ckpt_path}...', logger )
    # load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    optimizer.load_state_dict(state_dict['optimizer'])


def remodel_ckpt_for_adjusted_architectures(base_model, model_name, state_dict, arch=None):
    """
    This function remodels the checkpoint for adjusted architectures.
    It is used to load the weights from a checkpoint into a model with a different architecture.
    """
    base_ckpt = state_dict
    # remodel the ckpt for adjusted architecture
    if model_name == 'ScaphoidAdaPoinTr':

        if state_dict.get('model') is not None:
            base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['model'].items()}
        elif state_dict.get('base_model') is not None:
            base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
        else:
            raise RuntimeError('mismatch of ckpt weight')
        base_ckpt = {'module.'+k: v for k, v in base_ckpt.items()}  # valentino: key mismatch when module. is missing

        scaphoid_ckpt = base_ckpt.copy()

        volar_grouper = {k.replace(".grouper.", ".volar_proxies_encoder.grouper."): v for k, v in base_ckpt.items() if ".grouper." in k}
        dorsal_grouper = {k.replace(".grouper.", ".dorsal_proxies_encoder.grouper."): v for k, v in base_ckpt.items() if ".grouper." in k}
        volar_pos_embed = {k.replace(".pos_embed.", ".volar_proxies_encoder.pos_embed."): v for k, v in base_ckpt.items() if ".pos_embed." in k}
        dorsal_pos_embed = {k.replace(".pos_embed.", ".dorsal_proxies_encoder.pos_embed."): v for k, v in base_ckpt.items() if ".pos_embed." in k}
        volar_input_proj = {k.replace(".input_proj.", ".volar_proxies_encoder.input_proj."): v for k, v in base_ckpt.items() if ".input_proj." in k}
        dorsal_input_proj = {k.replace(".input_proj.", ".dorsal_proxies_encoder.input_proj."): v for k, v in base_ckpt.items() if ".input_proj." in k}
        everything_else = {k: v for k, v in base_ckpt.items() if ".grouper." not in k and ".pos_embed." not in k and ".input_proj." not in k}

        scaphoid_ckpt = {**everything_else, **volar_grouper, **dorsal_grouper, **volar_pos_embed, **dorsal_pos_embed, 
                         **volar_input_proj, **dorsal_input_proj}
        

        base_ckpt = scaphoid_ckpt
    
    elif model_name == 'ScaphoidPointAttN':

        print_log(f'Loading weights from {state_dict["net_state_dict"].keys()}...')

        base_ckpt = state_dict['net_state_dict']        # 'encoder.*' prefix of weights for feature extractor and seed generator. 'refine.*' for refine module, 'refine1.*' for refine1 module
        base_model_keys = base_model.state_dict().keys()

        
        
        base_ckpt1 = {k.replace("encoder.", "module.net_blocks.1."): v for k, v in base_ckpt.items() if "encoder." in k}   # keys of FE and SG cannot be distinguished because they have the same prefix 'encoder.'
        base_ckpt2 = {k.replace("encoder.", "module.net_blocks.2."): v for k, v in base_ckpt.items() if "encoder." in k}   # so instead create two separate dicts for each block
        base_ckpt_merge = {**base_ckpt1, **base_ckpt2}
        base_ckpt_fe_sg = {k : v for k, v in base_ckpt_merge.items() if k in base_model_keys}                 # and then delete the keys that are not in the model state dict

        base_ckpt_pg_0 = {k.replace("refine.", "module.net_blocks.3.refine."): v for k, v in base_ckpt.items() if "refine." in k}
        base_ckpt_pg_1 = {k.replace("refine1.", "module.net_blocks.3.refine1."): v for k, v in base_ckpt.items() if "refine1." in k}

        base_ckpt = {**base_ckpt_fe_sg, **base_ckpt_pg_0, **base_ckpt_pg_1}  # merge the two dicts

    elif model_name == 'CompletionPoseEstPointAttN':
        print_log(f'Loading weights from {state_dict["net_state_dict"].keys()}...')

        base_ckpt = state_dict['net_state_dict']
        base_model_keys = base_model.state_dict().keys()

        if 'concat' in arch:

            # Step 1: Re-map encoder weights to correct module names
            base_ckpt1 = {k.replace("encoder.", "module.net_blocks.5."): v for k, v in base_ckpt.items() if "encoder." in k}
            base_ckpt11 = {k.replace("encoder.", "module.net_blocks.6."): v for k, v in base_ckpt.items() if "encoder." in k}
            base_ckpt2 = {k.replace("encoder.", "module.net_blocks.8."): v for k, v in base_ckpt.items() if "encoder." in k}
            base_ckpt_merge = {**base_ckpt1, **base_ckpt2, **base_ckpt11}
            base_ckpt_fe_sg = {k: v for k, v in base_ckpt_merge.items() if k in base_model_keys}

            # Step 2: Re-map refine modules
            base_ckpt_pg_0 = {k.replace("refine.", "module.net_blocks.9.refine."): v for k, v in base_ckpt.items() if "refine." in k}
            base_ckpt_pg_1 = {k.replace("refine1.", "module.net_blocks.9.refine1."): v for k, v in base_ckpt.items() if "refine1." in k}

        elif 'affil' in arch:
            base_ckpt1 = {k.replace("encoder.", "module.net_blocks.6."): v for k, v in base_ckpt.items() if "encoder." in k}
            base_ckpt2 = {k.replace("encoder.", "module.net_blocks.7."): v for k, v in base_ckpt.items() if "encoder." in k}
            base_ckpt_merge = {**base_ckpt1, **base_ckpt2}
            base_ckpt_fe_sg = {k: v for k, v in base_ckpt_merge.items() if k in base_model_keys}

            # Step 2: Re-map refine modules
            base_ckpt_pg_0 = {k.replace("refine.", "module.net_blocks.8.refine."): v for k, v in base_ckpt.items() if "refine." in k}
            base_ckpt_pg_1 = {k.replace("refine1.", "module.net_blocks.8.refine1."): v for k, v in base_ckpt.items() if "refine1." in k}


        # Step 3: Rename keys for backward compatibility
        base_ckpt_fe_sg = ckpt_key_renamer_for_older_models(base_ckpt_fe_sg)
        base_ckpt_pg_0 = ckpt_key_renamer_for_older_models(base_ckpt_pg_0)
        base_ckpt_pg_1 = ckpt_key_renamer_for_older_models(base_ckpt_pg_1)

        # Step 4: Merge all processed checkpoints
        base_ckpt = {**base_ckpt_fe_sg, **base_ckpt_pg_0, **base_ckpt_pg_1}


    else:
        raise NotImplementedError(f"Model name {model_name} not supported for checkpoint remodeling.")

    return base_ckpt


def load_model(base_model, ckpt_path, model_name, logger = None, arch=None):
    if not os.path.exists(ckpt_path):
        raise NotImplementedError('no checkpoint file from path %s...' % ckpt_path)
    print_log(f'Loading weights from {ckpt_path}...', logger = logger)

    # load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        
    # remodel the ckpt for adjusted architecture
    base_ckpt = remodel_ckpt_for_adjusted_architectures(base_model, model_name, state_dict, arch=arch)

    model_dict = base_model.state_dict()
    for name, param in base_ckpt.items():
        if name in model_dict:
            if param.shape == model_dict[name].shape:
                # exact match → load as is
                model_dict[name].copy_(param)
            else:
                # shape mismatch → try to copy what we can
                tgt = model_dict[name]
                if param.ndim == tgt.ndim and all(s <= t for s, t in zip(param.shape, tgt.shape)):
                    print(f"Partial load for {name}: checkpoint {param.shape} → model {tgt.shape}")
                    # fill matching dimensions
                    try:
                        tgt[:param.shape[0], :param.shape[1], ...] = param
                    except:
                        tgt[:param.shape[0], ...] = param
                    # leave the rest of tgt unchanged (could also zero or randomize)
                else:
                    print(f"Skipping {name}: incompatible shapes {param.shape} vs {tgt.shape}")
        else:
            print(f"Skipping {name}: not found in model")

    model_keys = set(base_model.state_dict().keys())
    ckpt_keys = set(model_dict.keys())
    missing_keys = model_keys - ckpt_keys
    unexpected_keys = ckpt_keys - model_keys

    # if missing_keys:
    #     print_log(f"[WARNING] Missing keys in checkpoint: {missing_keys}", logger=logger)
    if unexpected_keys:
        print_log(f"[WARNING] Unexpected keys in checkpoint: {sorted(unexpected_keys)}", logger=logger, color='red')
        print_log(f"Required keys in model: {sorted(model_keys)}", logger=logger, color='yellow')
        raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected_keys}")
    else:
        print_log(f"[INFO] No unexpected keys in checkpoint.", logger=logger, color='green')


    base_model.load_state_dict(model_dict, strict=False)

    epoch = -1
    if state_dict.get('epoch') is not None:
        epoch = state_dict['epoch']
    if state_dict.get('metrics') is not None:
        metrics = state_dict['metrics']
        if not isinstance(metrics, dict):
            metrics = metrics.state_dict()
    else:
        metrics = 'No Metrics'
    print_log(f'ckpts @ {epoch} epoch( performance = {str(metrics):s})', logger = logger, color='green')
    return 


def save_checkpoint(base_model, optimizer, epoch, cdl1_score, best_cdl1_score, prefix, args, logger = None):
    """
    Adapted from PoinTr to use different print_log function
    """
    print_log(f"################### Saving Checkpoint: {prefix} #####################", logger, color='red')
    if args.local_rank == 0 and not args.debug and not args.smac3:
        torch.save({
                    'base_model' : base_model.module.state_dict() if args.distributed else base_model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'epoch' : epoch,
                    'metrics' : {'CDL1': cdl1_score} if cdl1_score is not None else dict(),
                    'best_metrics' : {'CDL1': best_cdl1_score} if best_cdl1_score is not None else dict(),
                    }, os.path.join(args.experiment_path, prefix + '.pth'))
        print_log(f"Save checkpoint at {os.path.join(args.experiment_path, prefix + '.pth')}", logger = logger)
    else:
        print_log(f"Skip saving checkpoint because of Debug={args.debug} and SMAC3={args.smac3} at \
                  {os.path.join(args.experiment_path, prefix + '.pth')}", logger = logger, color='yellow')