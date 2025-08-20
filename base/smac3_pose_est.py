import os
import sys
import argparse

from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, CategoricalHyperparameter
from ConfigSpace.hyperparameters import UniformFloatHyperparameter as UniformFloatHyp
from smac import Scenario
from smac.facade import MultiFidelityFacade
import torch


current_dir = os.path.dirname(os.path.abspath(__file__))
main_dir = os.path.abspath(os.path.join(current_dir, ".."))
poin_tr_dir = os.path.join(main_dir, 'submodules', 'PoinTr')

# Add the main directory to sys.path such that submodules can be imported
if main_dir not in sys.path:
    sys.path.append(main_dir)
    sys.path.append(poin_tr_dir)


from base.main_pose_est_both import main as main_pose_est_both
from base.main_pose_est import main as main_pose_est
from base.main import main as main_completion
from base.scaphoid_models.ScaphoidModules import set_dropout

# config_file = 'Scaphoid_models/PoseEstCfgs/PoseEst_ScaphoidPointAttN_both.yaml'
# project_postfix = 'PoseEst_both'
# config_folder = '/home/valantano/mt/repository/base/cfgs/'


def main():

    parser = argparse.ArgumentParser(description='Run SMAC3 Optimization')
    parser.add_argument('--config_file', default=None,
                        help=f'Path to config file (default: {None})')
    parser.add_argument('--config_folder', default=None,
                        help=f'Path to config folder (default: {None})')
    parser.add_argument('--project_postfix', default=None,
                        help=f'Postfix for the project (default: {None})')
    args = parser.parse_args()

    if not args.config_file or not args.config_folder or not args.project_postfix:
        raise ValueError("Please provide --config_file, --config_folder, and --project_postfix arguments.")
    
    global config_file, config_folder, project_postfix, main_func
    config_file = args.config_file
    config_folder = args.config_folder
    project_postfix = args.project_postfix
    if project_postfix == 'PoseEst_both':
        main_func = main_pose_est_both
    elif project_postfix == 'PoseEst_volar' or project_postfix == 'PoseEst_dorsal':
        main_func = main_pose_est
    elif project_postfix == 'Static_volar' or project_postfix == 'Static_dorsal' or project_postfix == 'Static_max':
        main_func = main_completion
    else:
        raise ValueError(f"Unknown project_postfix: {project_postfix}. Expected 'PoseEst_both', 'PoseEst_volar', \
                         'PoseEst_dorsal', 'Static_volar', or 'Static_dorsal'.")

    cs = ConfigurationSpace()

    cs.add(UniformFloatHyp("optimizer.lr", lower=1e-5, upper=1e-2, default_value=0.0001, log=True))
    cs.add(UniformFloatHyp("optimizer.weight_decay", lower=1e-6, upper=1e-2, default_value=0.0005, log=True))
    cs.add(UniformFloatHyp("model.dropout", lower=0.0, upper=0.5, default_value=0.0))

    cs.add(UniformIntegerHyperparameter("scheduler.decay_step", lower=5, upper=50, default_value=21))
    cs.add(UniformFloatHyp("scheduler.lr_decay", lower=0.7, upper=0.99, default_value=0.9))
    cs.add(UniformFloatHyp("scheduler.lowest_decay", lower=0.001, upper=0.1, default_value=0.02, log=True))

    cs.add(UniformIntegerHyperparameter("bnmscheduler.decay_step", lower=5, upper=50, default_value=21))
    cs.add(UniformFloatHyp("bnmscheduler.bn_decay", lower=0.1, upper=0.9, default_value=0.5))
    cs.add(UniformFloatHyp("bnmscheduler.bn_momentum", lower=0.7, upper=0.99, default_value=0.9))
    cs.add(UniformFloatHyp("bnmscheduler.lowest_decay", lower=0.001, upper=0.1, default_value=0.01, log=True))

    cs.add(CategoricalHyperparameter("total_bs", [8, 16, 32, 64], default_value=32))  # power-of-2 steps
    cs.add(CategoricalHyperparameter("step_per_update", [1, 2, 4], default_value=1))

    # scenario = Scenario(
    #     configspace=cs,
    #     name="PoseEstPointAttN_Both",
    #     output_directory=f"./smac_results_{project_postfix}",
    #     deterministic=False,
    #     n_trials=5,
    #     use_default_config=True,
    #     min_budget=2,  # Min epochs
    #     max_budget=3,  # Max epochs
    #     seed=0,
    # )
    scenario = Scenario(
        configspace=cs,
        name="PoseEstPointAttN_Both",
        output_directory=f"./smac_results_{project_postfix}",
        deterministic=False,
        n_trials=60,
        use_default_config=True,
        min_budget=30,  # Min epochs
        max_budget=100,  # Max epochs
        seed=0,
    )

    smac = MultiFidelityFacade(scenario, objective, overwrite=False)

    incumbent = smac.optimize()



def objective(config, budget, seed, fidelity=None):
    """
    Objective function for SMAC optimization.
    :param config: Configuration from SMAC.
    :param budget: Current budget (epoch).
    :param fidelity: Not used, but required by SMAC.
    :return: Validation loss.
    """
    global config_file, config_folder, project_postfix, main_func
    priority_args = {
        # Core SMAC3 settings
        'smac3': True,
        'debug': False,
        'val_freq': max(1, int(budget // 5)),  # Validate every 1/5 of budget
        'exp_name': f'smac3_run_{hash(str(config))}{seed}',  # Unique experiment name
        
        # All other arguments from your namespace - fixed values
        'config': config_file,
        # 'config_folder': '/home/fn848825/multi-view-point-cloud-completion/base/cfgs/',
        'config_folder': config_folder,
        'launcher': 'none',
        'local_rank': 0,
        'num_workers': 4,
        'seed': seed,
        'deterministic': False,
        'sync_bn': False,
        'start_ckpts': None,
        'ckpts': None,
        'resume': False,
        'test': False,
        'mode': None,
        'pretrained': None,
        'strict': False,
        'partly': None,
        'wandb': 'Master Thesis SMAC3' + project_postfix  # WandB project name
    }

    priority_config = {
        'max_epoch': int(budget),
        'total_bs': config['total_bs'],
        'step_per_update': config['step_per_update'],
        
        # Optimizer settings
        'optimizer': {
            'kwargs': {
                'lr': config['optimizer.lr'],
                'weight_decay': config['optimizer.weight_decay']
            }
        },
        
        # Scheduler settings
        'scheduler': {
            'kwargs': {
                'decay_step': config['scheduler.decay_step'],
                'lr_decay': config['scheduler.lr_decay'],
                'lowest_decay': config['scheduler.lowest_decay']
            }
        },
        
        # BN momentum scheduler settings
        'bnmscheduler': {
            'kwargs': {
                'decay_step': config['bnmscheduler.decay_step'],
                'bn_decay': config['bnmscheduler.bn_decay'],
                'bn_momentum': config['bnmscheduler.bn_momentum'],
                'lowest_decay': config['bnmscheduler.lowest_decay']
            }
        }
    }


    # Train the model
    set_dropout(config['model.dropout'])  # Set dropout rate
    try:
        best_cdl1_score_val: torch.Tensor = main_func(
            priority_config=priority_config,
            priority_args=priority_args
        )
    except Exception as e:
        print(f"Error during training: {e}")
        best_cdl1_score_val = float('inf')

    if isinstance(best_cdl1_score_val, torch.Tensor):
        best_cdl1_score_val = best_cdl1_score_val.detach().cpu().numpy().item()

    return best_cdl1_score_val


if __name__ == '__main__':
    main()