import os
from pathlib import Path

import yaml
from easydict import EasyDict

from base.scaphoid_utils.logger import print_log


class ConfigHandler:
    """
    Used to load a config file.
    """

    def __init__(self, cfg_folder, resume = False, logger = None):
        self.cfg_folder = cfg_folder
        self.resume = resume
        self.logger = logger

    def get_abs_path(self, relative_path, resume = False):
        if resume:
            return relative_path
        else:
            return Path(os.path.join(self.cfg_folder, relative_path)).resolve()

    def merge_new_config(self, config, new_config):
        """
        Go through yaml file and if a key with _base_ is found, 
        load the base yaml file and merge it with the current yaml file
        """
        for key, val in new_config.items():
            if not isinstance(val, dict):
                if key == '_base_':
                    with open(self.get_abs_path(new_config['_base_']), 'r') as f:
                        try:
                            val = yaml.load(f, Loader=yaml.FullLoader)
                        except:
                            val = yaml.load(f)
                    config[key] = EasyDict()
                    self.merge_new_config(config[key], val)
                else:
                    config[key] = val
                    continue
            if key not in config:
                config[key] = EasyDict()
            self.merge_new_config(config[key], val)
        return config

    def cfg_from_yaml_file(self, cfg_file):
        config = EasyDict()
        print_log(f"Loading yaml file from {self.get_abs_path(cfg_file, resume=self.resume)}", 
                  self.logger, color='blue')
        with open(self.get_abs_path(cfg_file, resume=(self.resume)), 'r') as f:
            try:
                new_config = yaml.load(f, Loader=yaml.FullLoader)
            except:
                new_config = yaml.load(f)
        self.merge_new_config(config=config, new_config=new_config)
        return config
    
    def load_yaml_file(self, cfg_file):
        with open(self.get_abs_path(cfg_file, resume=self.resume), 'r') as f:
            try:
                new_config = yaml.load(f, Loader=yaml.FullLoader)
            except:
                new_config = yaml.load(f)
        return new_config

    def get_config(self, args):
        if args.resume or args.test:
            cfg_path = os.path.join(args.experiment_path, 'config.yaml')
            if not os.path.exists(cfg_path):
                print_log(f"Failed to resume {os.path.abspath(cfg_path)}", logger = self.logger, color='red')
                raise FileNotFoundError(f"Failed to resume {os.path.abspath(cfg_path)}")
            print_log(f'Resume yaml from {cfg_path}', logger = self.logger, color='blue')
            args.config = cfg_path

        config = self.cfg_from_yaml_file(args.config)
        if not self.resume and args.local_rank == 0 and not args.test:
            self.save_experiment_config(args, config)

        if 'transform_with' not in config.dataset:
            raise ValueError('transform_with not in config')
        if 'transforms' not in config.dataset:
            raise ValueError('transform not in config')
        if 'reverse_transforms' not in config.dataset:
            raise ValueError('reverse_transforms not in config')
        if 'model' not in config:
            raise ValueError('model not in config')
        return config
    
    def save_experiment_config(self, args, config):
        """
        @param args: arguments from the command line
        @param config: configuration dictionary
        """
        config_path = os.path.join(args.experiment_path, 'config.yaml')
        # with open(config_path, 'w') as f:
        #     yaml.dump(config, f)
        if not args.debug:
            os.system('cp %s %s' % (self.get_abs_path(args.config, resume=self.resume), config_path))
            print_log(f'Copy the Config file from {args.config} to {config_path}',logger = self.logger, color='blue')