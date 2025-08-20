import time, logging

from easydict import EasyDict

from submodules.PoinTr.utils.logger import get_logger, logging


def get_table_str(*rows: list[str]):
    # Transpose to get columns
    columns = list(zip(*rows))
    # Find max width for each column
    col_widths = [max(len(str(cell)) for cell in col) for col in columns]
    # Build each row with padded columns
    lines = []
    for row in rows:
        padded = [str(cell).ljust(width) for cell, width in zip(row, col_widths)]
        lines.append("  ".join(padded))
    return "\n".join(lines), lines


class DebugLogger(logging.Logger):

    def __init__(self):
        pass

    def log(self, msg, level=logging.INFO):
        print(msg)


class Logger:

    def __init__(self, experiment_path, log_name="default", debug=False):
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = os.path.join(experiment_path, f'{timestamp}.log')
        self.debug = debug
        if not self.debug:
            self.logger = get_logger(name=log_name, log_file=log_file, log_level=logging.INFO)
        else:
            self.logger = DebugLogger()
        # logging_filter = logging.Filter(log_name)
        # logging_filter.filter = lambda record: record.find(log_name) != -1

        self.nested_level = 0
        self.nest_symbol = '---'

    def log(self, message, level=logging.INFO):
        self.logger.log(msg=f"{self.nest_symbol*self.nested_level}{message}", level=level)

    def log_model_info(self, base_model):
        print_log('Trainable_parameters:', self)
        print_log('=' * 25, self)
        for name, param in base_model.named_parameters():
            if param.requires_grad:
                print_log(name, self)
        print_log('=' * 25, self)
        
        print_log('Untrainable_parameters:', self)
        print_log('=' * 25, self)
        for name, param in base_model.named_parameters():
            if not param.requires_grad:
                print_log(name, self)
        print_log('=' * 25, self)

    def log_args(self, args, pre='args'):
        """
        Adapted from from submodules.PoinTr.utils.config import log_args_to_file
        """
        print_log(f"################### Logging args #####################", self, color='blue')
        for key, val in args.__dict__.items():
            print_log(f'---{pre}.{key} : {val}', logger=self)

    def log_config(self, cfg, pre='cfg'):
        """
        Adapted from from submodules.PoinTr.utils.config import log_config_to_file
        """
        if pre == 'cfg' or pre == 'config':
            print_log(f"################### Logging config #####################", self, color='blue')
        else:
            pre = '-' * len(pre)
        for key, val in cfg.items():
            if isinstance(cfg[key], EasyDict):
                print_log(f'{pre}.{key} = edict()', logger=self)
                self.log_config(cfg[key], pre=pre + '.' + key)
                continue
            print_log(f'{pre}.{key} : {val}', logger=self)


class LoggerLevel(Logger):
    """
    Logger with a level that can be increased and decreased.
    Used to temporarily increase the log level for a specific block of code.
    """

    def __init__(self, logger: Logger, nested_level=0, nest_symbol='---', debug=False):
        self.logger = logger
        self.nested_level = nested_level
        self.debug = debug
        self.nest_symbol = nest_symbol

    def log(self, message, level=None):
        if level is None:
            level = self.level
        super().log(message, level)


color_map = {
        'blue': "\033[94m",  # Blue
        'green': "\033[92m",  # Green
        'yellow': "\033[93m",  # Yellow
        'red': "\033[91m",  # Red
        'magenta': "\033[95m",  # Magenta
        'cyan': "\033[96m",  # Cyan
        'white': "\033[0m",  # Reset
    }

def print_log(msg, logger: Logger=None, level=logging.INFO, color='white'):
    """
    Adapted from submodules.PoinTr.utils.logger import print_log
    Print a log message.
    :param msg: The message to be logged.
    :param logger: logger to be used, either a Logger object or None.
    :param level: Logging level. Only available when `logger` is a Logger
    :param color: Color of the message. Default is 'white'.
    """
    msg = f"{color_map[color]}{msg}{color_map['white']}"
    if logger is None:
        print(msg)
    elif isinstance(logger, Logger):
        logger.log(msg, level)
    else:
        raise TypeError(f'logger should be either a logging.Logger object, or None, but got {type(logger)}')


from abc import ABC, abstractmethod
import os
class MetricLogger(ABC):
    """
    Abstract class for logging metrics.
    Inherited by SimulLogger, TensorboardLogger and WandbLogger (see belwo)
    """

    @abstractmethod
    def log_batch_metrics(self, metrics: dict, step: int, mode='train'):
        pass

    @abstractmethod
    def log_epoch_metrics(self, metrics: dict, epoch: int, mode='train'):
        pass

    @abstractmethod
    def add_image(self, name, input_pc, epoch, mode='val'):
        pass

    @abstractmethod
    def add_input_3d(self, name, gt, input_used: list, epoch, mode='val'):
        pass

    @abstractmethod
    def add_pred_3d(self, name, gt, pred, epoch, mode='val'):
        pass

    @abstractmethod
    def close(self):
        pass

class SimulLogger(MetricLogger):
    """
    Simultaneously log metrics to tensorboard and wandb using the TensorboardLogger and WandbLogger classes
    """

    def __init__(self, args, config):
        self.tensorboad_logger = TensorboardLogger(args, config)

        self.wandb_logger = None
        if not args.debug:# and not args.test:
            self.wandb_logger = WandbLogger(args, config)

        

    def log_batch_metrics(self, metrics: dict, step: int, mode='train'):
        """
        Log batch metrics only to tensorboard (not wandb because of sensible data)
        :param metrics: dictionary of metrics   {'CDL1': 0.1, 'CDL2': 0.2, ...}
        :param step: current step
        :param mode: train or val or test
        """
        self.tensorboad_logger.log_batch_metrics(metrics, step, mode)
        # self.wandb_logger.log_batch_metrics(metrics, step, mode)
        
            
    def log_epoch_metrics(self, metrics: dict, epoch: int, mode='train'):
        """
        Log epoch metrics to tensorboard and wandb
        :param metrics: dictionary of metrics   {'CDL1': 0.1, 'CDL2': 0.2, ...}
        :param epoch: current epoch
        :param mode: train or val or test
        """
        self.tensorboad_logger.log_epoch_metrics(metrics, epoch, mode)
        if self.wandb_logger:
            self.wandb_logger.log_epoch_metrics(metrics, epoch, mode)

    def add_image(self, name, pc_image, epoch, mode='val'):
        """
        Log point cloud image to tensorboard (not wandb because of sensible data)
        :param name: name of the image
        :param pc_image: point cloud image [H, W, 3]
        :param epoch: current epoch
        :param mode: train or val or test
        """
        self.tensorboad_logger.add_image(name, pc_image, epoch, mode)

    def add_input_3d(self, gt, input_used: list, epoch, mode='val'):
        """
        Log input point clouds to tensorboard (not wandb because of sensible data)
        :param gt: ground truth points [B, N, 3] with B=1
        :param partial_volar: partial volar points [B, N, 3] with B=1
        :param partial_dorsal: partial dorsal points [B, N, 3] with B=1
        :param epoch: current epoch
        :param mode: train or val or test
        """
        self.tensorboad_logger.add_input_3d(gt, input_used, epoch, mode)

    def add_pred_3d(self, gt, pred, epoch, mode='val'):
        """
        Log predicted point clouds to tensorboard (not wandb because of sensible data)
        :param gt: ground truth points [B, N, 3] with B=1
        :param pred: predicted points [B, N, 3] with B=1
        :param epoch: current epoch
        :param mode: train or val or test
        """
        self.tensorboad_logger.add_pred_3d(gt, pred, epoch, mode)


    def close(self):
        """
        Close the tensorboard and wandb loggers
        """
        self.tensorboad_logger.close()
        if self.wandb_logger:
            self.wandb_logger.close()

from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torch import tensor
import base.scaphoid_utils.constants as const
class TensorboardLogger(MetricLogger):

    def __init__(self, args, config):
        self.train_writer = SummaryWriter(os.path.join(args.tfboard_path, 'train'))
        self.val_writer = SummaryWriter(os.path.join(args.tfboard_path, 'val'))
        self.test_writer = SummaryWriter(os.path.join(args.tfboard_path, 'test'))

    def log_batch_metrics(self, metrics: dict, step: int, mode='train'):
        writer = self.train_writer if mode == 'train' else self.val_writer if mode == 'val' else self.test_writer
        for key, value in metrics.items():
            writer.add_scalar(key, value, step)

    def log_epoch_metrics(self, metrics: dict, epoch: int, mode='train'):
        writer = self.train_writer if mode == 'train' else self.val_writer if mode == 'val' else self.test_writer
        for key, value in metrics.items():
            writer.add_scalar(key, value, epoch)

    def add_image(self, name, pc_image, epoch, mode='val'):
        """
        :param name: name of the image
        :param pc_image: point cloud image [H, W, 3]
        :param epoch: current epoch
        :param mode: train or val or test
        """
        writer = self.train_writer if mode == 'train' else self.val_writer
        writer.add_image(name , pc_image, epoch, dataformats='HWC')

    def add_input_3d(self, name, gt, input_used: tensor, epoch, mode='val'):
        """
        :param gt: ground truth points [B, N, 3] with B=1
        :param input_used: partial points [B, N, 3] with B=1
        :param epoch: current epoch
        :param mode: train or val or test
        """
        writer = self.train_writer if mode == 'train' else self.val_writer if mode == 'val' else self.test_writer
        gt, partial = gt.squeeze(), input_used.squeeze()

        gt_colors = np.repeat(const.gt_unfocused_rgb[np.newaxis, :], gt.shape[0], axis=0)
        partial_colors = np.repeat(const.partial_volar_rgb[np.newaxis, :], partial.shape[0], axis=0)

        verticies = np.array([np.concatenate((gt, partial, ), axis=0)])
        colors = np.array([np.concatenate((gt_colors, partial_colors, ), axis=0)])

        # point_size_config = {
        #     'material': {
        #         'cls': 'PointsStandardMaterial',
        #         'size': 2.0
        #     }
        # }
        
        writer.add_mesh(name, vertices=tensor(verticies), colors=tensor(colors), global_step=epoch)
    
    def add_pred_3d(self, name, gt, pred, epoch, mode='val'):
        """
        :param gt: ground truth points [B, N, 3] with B=1
        :param pred: predicted points [B, N, 3] with B=1
        :param epoch: current epoch
        :param mode: train or val or test
        """
        # print(f"add_pred_3d: {name}, {gt.shape}, {pred.shape}, {epoch}, {mode}")
        writer = self.train_writer if mode == 'train' else self.val_writer if mode == 'val' else self.test_writer
        gt_colors = np.repeat(const.gt_unfocused_rgb[np.newaxis, :], gt.shape[0], axis=0)
        pred_colors = np.repeat(const.pred_rgb[np.newaxis, :], pred.shape[0], axis=0)

        verticies = np.concatenate((gt, pred), axis=0)
        colors = np.concatenate((gt_colors, pred_colors), axis=0)

        writer.add_mesh(name+'GT', vertices=tensor(verticies).unsqueeze(0), colors=tensor(colors).unsqueeze(0), 
                        global_step=epoch)
        writer.add_mesh(name, vertices=tensor(pred).unsqueeze(0), colors=tensor(pred_colors).unsqueeze(0), 
                        global_step=epoch)


    def close(self):
        self.train_writer.close()
        self.val_writer.close()

import wandb
class WandbLogger(MetricLogger):

    def __init__(self, args, config):

        wandb_run_id = None
        if args.resume:
            run_id_path = os.path.join(args.experiment_path, 'wandb_id.txt')
            if not os.path.exists(run_id_path):
                raise FileNotFoundError(f'wandb_id.txt not found in {args.experiment_path}')
            with open(run_id_path, 'r') as f:
                wandb_run_id = f.read().strip()

        config_dict = None
        if config is not None:
            config_dict = dict(config)
            
        run = wandb.init(
            # entity=args.exp_name, # name of me or my team
            name = args.exp_name,
            project=args.wandb,  # name of the project - default is "Master Thesis"
            config=config_dict,
            id=wandb_run_id,
            resume="allow"
        )
        self.run = run

        if not args.resume:
            with open(os.path.join(args.experiment_path, 'wandb_id.txt'), 'w') as f:
                f.write(run.id)

        wandb.define_metric("train/Loss/Batch*", step_metric="batch/step")
        wandb.define_metric("val/Loss/Epoch", step_metric="epoch/step")
        wandb.define_metric("test/Loss/Epoch", step_metric="epoch/step")

    def log_batch_metrics(self, metrics: dict, step: int, mode='train'):
        prefixed_metrics = {f"{mode}/{key}": value for key, value in metrics.items()}
        prefixed_metrics['batch/step'] = step

        wandb.log(prefixed_metrics)

    def log_epoch_metrics(self, metrics: dict, epoch: int, mode='train'):
        prefixed_metrics = {f"{mode}/{key}": value for key, value in metrics.items()}
        prefixed_metrics['epoch/step'] = epoch

        wandb.log(prefixed_metrics)

    # leave empty for now, since it is not used because the sensible data should not be uploaded to wandb
    def add_image(self, name, input_pc, epoch, mode='val'):
        pass

    def add_input_3d(self, gt, partial_volar, partial_dorsal, epoch, mode='val'):
        pass

    def add_pred_3d(self, gt, pred, epoch, mode='val'):
        pass

    def close(self):
        wandb.finish()