import os
import glob
import torch
import traceback
from lib.train.admin import multigpu
from torch.utils.data.distributed import DistributedSampler

class BaseTrainer:
    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None):
        """
        args:
            actor - The actor for training the network
            loaders - list of dataset loaders, e.g. [train_loader, val_loader]. In each epoch, the trainer runs one
                        epoch for each loader.
            optimizer - The optimizer used for training, e.g. Adam
            settings - Training settings
            lr_scheduler - Learning rate scheduler
        """
        self.actor = actor
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loaders = loaders

        self.update_settings(settings)

        self.epoch = 0
        self.stats = {}

        self.device = getattr(settings, 'device', None)
        if self.device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() and settings.use_gpu else "cpu")

        self.actor.to(self.device)
        self.settings = settings

    def update_settings(self, settings=None):
        """Updates the trainer settings. Must be called to update internal settings."""
        if settings is not None:
            self.settings = settings

        if self.settings.env.workspace_dir is not None:
            self.settings.env.workspace_dir = os.path.expanduser(self.settings.env.workspace_dir)
            '''2021.1.4 New function: specify checkpoint dir'''
            if self.settings.save_dir is None:
                self._checkpoint_dir = os.path.join(self.settings.env.workspace_dir, 'checkpoints')
            else:
                self._checkpoint_dir = os.path.join(self.settings.save_dir, 'checkpoints')
            print(f"\033[93mcheckpoints will be saved to:\033[0m {self._checkpoint_dir}")

            if self.settings.local_rank in [-1, 0]:
                if not os.path.exists(self._checkpoint_dir):
                    print(f"\033[91mcheckpoints directory doesn't exist. "
                          f"Create checkpoints directory\033[0m")
                    os.makedirs(self._checkpoint_dir)
        else:
            self._checkpoint_dir = None

    def train(self, max_epochs, load_latest=False, load_pretrained_ckpt=False):
        """Do training for the given number of epochs.
        args:
            max_epochs - Max number of training epochs,
            load_latest - Bool indicating whether to resume from latest epoch.
            load_pretrained_ckpt - Bool indicating whether to load pretrained weights.
        """

        if load_latest and load_pretrained_ckpt:
            raise ValueError('load_latest and load_pretrained_ckpt cannot be True at the same time.')
        try:
            if load_latest:  # 断点重训, 恢复optimizer
                self.load_checkpoint()
            if load_pretrained_ckpt:  # 加载预训练权重
                self.load_state_dict(self.settings.env.pretrained_ckpt_dir)

            for epoch in range(self.epoch+1, max_epochs+1):
                self.epoch = epoch

                self.train_epoch()

                if self.lr_scheduler is not None:
                    if self.settings.scheduler_type != 'cosine':
                        self.lr_scheduler.step()
                    else:
                        self.lr_scheduler.step(epoch - 1)

                # save ckpt if condition is met
                save_every_n_epoch = getattr(self.settings, "save_every_n_epoch", 10)
                save_last_n_ckpt = getattr(self.settings, "save_last_n_ckpt", 10)
                save_ckpt_list = getattr(self.settings, "save_ckpt_list", [])

                if epoch > (max_epochs - save_last_n_ckpt) or epoch % save_every_n_epoch == 0 or epoch in save_ckpt_list:
                    if self._checkpoint_dir:
                        if self.settings.local_rank in [-1, 0]:
                            self.save_checkpoint()
        except Exception as e:
            raise RuntimeError(f'Training crashed at epoch {self.epoch}') from e

        print('\033[92mFinished training!\033[0m')

    def train_epoch(self):
        raise NotImplementedError('train_epoch must be implemented in the subclass.')

    def save_checkpoint(self):
        net = self.actor.net.module if multigpu.is_multi_gpu(self.actor.net) else self.actor.net

        actor_type = type(self.actor).__name__
        net_type = type(net).__name__
        state = {
            'epoch': self.epoch,
            'actor_type': actor_type,
            'net_type': net_type,
            'net': net.state_dict(),
            'net_info': getattr(net, 'info', None),
            'constructor': getattr(net, 'constructor', None),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
            'stats': self.stats,
            'settings': self.settings
        }

        directory = f'{self._checkpoint_dir}'
        if not os.path.exists(directory):
            print("\033[91mcheckpoint directory doesn't exist when trying to save. creating...\033[0m")
            os.makedirs(directory)

        # First save as a tmp file
        tmp_file_path = f'{directory}/{net_type}_ep{self.epoch:04d}.tmp'
        torch.save(state, tmp_file_path)

        file_path = f'{directory}/{net_type}_ep{self.epoch:04d}.ckpt'

        # Now rename to actual checkpoint. os.rename seems to be atomic if files are on same filesystem. Not 100% sure
        os.rename(tmp_file_path, file_path)


    def load_checkpoint(self, checkpoint = None, fields = None, ignore_fields = None, load_constructor = False):
        """Loads a network checkpoint file.

        Can be called in three different ways:
            load_checkpoint():
                Loads the latest epoch from the workspace. Use this to continue training.
            load_checkpoint(epoch_num):
                Loads the network at the given epoch number (int).
            load_checkpoint(path_to_checkpoint):
                Loads the file from the given absolute path (str).
        """

        net = self.actor.net.module if multigpu.is_multi_gpu(self.actor.net) else self.actor.net

        net_type = type(net).__name__

        if checkpoint is None:
            # Load most recent checkpoint
            checkpoint_list = sorted(glob.glob(f'{self._checkpoint_dir}/{net_type}_ep*.ckpt'))
            if checkpoint_list:
                checkpoint_path = checkpoint_list[-1]
            else:
                raise FileNotFoundError(f'No checkpoint found in {self._checkpoint_dir} '
                                        f'matching {net_type}_ep*.ckpt, can not load checkpoint.')

        elif isinstance(checkpoint, int):
            # Checkpoint is the epoch number
            checkpoint_path = f'{self._checkpoint_dir}/{net_type}_ep{checkpoint:04d}.ckpt'

        elif isinstance(checkpoint, str):
            # checkpoint is the path
            if os.path.isdir(checkpoint):
                checkpoint_list = sorted(glob.glob(f'{checkpoint}/*_ep*.ckpt'))
                if checkpoint_list:
                    checkpoint_path = checkpoint_list[-1]
                else:
                    raise FileNotFoundError(f'No checkpoint found in {checkpoint}, can not load checkpoint.')
            else:
                checkpoint_path = os.path.expanduser(checkpoint)
        else:
            raise TypeError(f'checkpoint must be None, int or str, got {type(checkpoint)}')

        # Load network
        checkpoint_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        if net_type != checkpoint_dict['net_type']:
            raise ValueError(f'Network type mismatch: current={net_type},checkpoint = {checkpoint_dict["net_type"]}')

        if fields is None:
            fields = checkpoint_dict.keys()
        if ignore_fields is None:
            ignore_fields = ['settings']

        # Never load the scheduler. It exists in older checkpoints.
        ignore_fields.extend(['constructor', 'net_type', 'actor_type', 'net_info'])

        # Load all fields
        for key in fields:
            if key in ignore_fields:
                continue
            if key == 'model' or key == 'net':
                net.load_state_dict(checkpoint_dict[key], strict=True)
            elif key == 'optimizer':
                self.optimizer.load_state_dict(checkpoint_dict[key])
            elif key == 'lr_scheduler' and self.lr_scheduler is not None :
                if checkpoint_dict[key] is not None:
                    self.lr_scheduler.load_state_dict(checkpoint_dict[key])
            else:
                setattr(self, key, checkpoint_dict[key])

        # Set the net info
        if load_constructor and 'constructor' in checkpoint_dict and checkpoint_dict['constructor'] is not None:
            net.constructor = checkpoint_dict['constructor']
        if 'net_info' in checkpoint_dict and checkpoint_dict['net_info'] is not None:
            net.info = checkpoint_dict['net_info']

        if 'epoch' in fields:
            for loader in self.loaders:
                if hasattr(loader.sampler, "set_epoch"):
                    loader.sampler.set_epoch(self.epoch)
        print(f'\033[93mCheckpoint successfully loaded from\033[0m {checkpoint_path} \033[93mat epoch\033[0m {self.epoch}\n')
        print(f'\033[93moptimizer state:\033[0m:')
        print("\t\033[93moptimizer type:\033[0m", type(self.optimizer).__name__)
        for i, group in enumerate(self.optimizer.param_groups):
            print(f"\tgroup {i}:")
            for k, v in group.items():
                if k == "params":
                    print(f"\t\033[93mparams:\033[0m {len(v)} tensors")
                else:
                    print(f"\t\033[93m{k}:\033[0m {v}")
        print(f'\033[93mlr scheduler state:\033[0m:')
        print("\t\033[93mscheduler type:\033[0m", type(self.lr_scheduler).__name__)
        print("\t\033[93mcurrent lr:\033[0m", self.lr_scheduler.get_last_lr())
        print("\t\033[93mstate_dict:\033[0m", self.lr_scheduler.state_dict(), '\n')
        return True

    def load_state_dict(self, pretrained_ckpt=None):
        """only load the network weights, not the optimizer or lr_scheduler.
        Used for loading pretrained weights."""

        net = self.actor.net.module if multigpu.is_multi_gpu(self.actor.net) else self.actor.net

        net_type = type(net).__name__
        if isinstance(pretrained_ckpt, str):
            # checkpoint is the path
            if os.path.isdir(pretrained_ckpt):
                checkpoint_list = sorted(glob.glob(f'{pretrained_ckpt}/*ep*.ckpt'))
                if checkpoint_list:
                    checkpoint_path = checkpoint_list[-1]
                else:
                    raise FileNotFoundError(f'No pretrained checkpoint found in {pretrained_ckpt}, can not load checkpoint.')
            else:
                checkpoint_path = os.path.expanduser(pretrained_ckpt)
        else:
            print(f'\033[91mpretrained_ckpt_path is not provided in settings, loading pretrained weights using cgf.PRETRINED_FILE_NAME.\033[0m')

            if not self.actor.cfg.TRAIN.PRETRAINED_FILE_NAME:
                raise RuntimeError('PRETRAINED_FILE_NAME must be set when pretrained_ckpt is not provided')
            current_dir = Path(__file__).resolve().parent  # ./trainer
            pretrained_path = current_dir.parents[2] / 'pretrained_models'
            checkpoint_path = pretrained_path / self.actor.cfg.TRAIN.PRETRAINED_FILE_NAME
            if not checkpoint_path.is_file():
                raise RuntimeError(f'Pretrained file not found: {checkpoint_path}')

        # Load network
        print(f"\033[93mLoading pretrained_ckpt from:\033[0m {checkpoint_path}")
        checkpoint_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        if net_type != checkpoint_dict['net_type']:
            raise ValueError(f'Network type mismatch: current={net_type},pretrained_checkpoint={checkpoint_dict["net_type"]}')

        if "model" in checkpoint_dict:
            model_name = 'model'
        elif "net" in checkpoint_dict:
            model_name = 'net'

        if hasattr(net, "backbone"):
            missing_k_1, unexpected_k_1 = net.backbone.load_state_dict(
                {k.removeprefix("backbone."): v
                 for k, v in checkpoint_dict[model_name].items()
                 if k.startswith("backbone.")},
                strict=False
            )
        else:
            raise RuntimeError(f"\033[91mError: net does not have backbone, cannot load pretrained weights for it.\033[0m")

        if hasattr(net, "multi_timescale_module"):
            missing_k_2, unexpected_k_2 = net.multi_timescale_module.load_state_dict(
                {k.removeprefix("multi_timescale_module."): v
                 for k, v in checkpoint_dict[model_name].items()
                 if k.startswith("multi_timescale_module.")},
                strict=False)
        else:
            print(f"\033[91mWarning: net does not have multi_timescale_module, skipping loading weights for it.\033[0m")
            missing_k_2, unexpected_k_2 = [], []

        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        print(f"\033[93mpretrained_ckpt is loaded from:\033[0m {checkpoint_path}")

        if missing_k_1:
            print(f"\033[91mbackbone missing keys:\033[0m {missing_k_1}")
        if unexpected_k_1:
            print(f"\033[91mbackbone unexpected keys:\033[0m {unexpected_k_1}")

        if missing_k_2:
            print(f"\033[91mmulti_timescale_module missing keys:\033[0m {missing_k_2}")
        if unexpected_k_2:
            print(f"\033[91mmulti_timescale_module unexpected keys:\033[0m{unexpected_k_2}")

        print(f'\n\033[93moptimizer state:\033[0m:')
        print("\t\033[93moptimizer type:\033[0m", type(self.optimizer).__name__)
        for i, group in enumerate(self.optimizer.param_groups):
            print(f"\tgroup {i}:")
            for k, v in group.items():
                if k == "params":
                    print(f"\t\033[93mparams:\033[0m {len(v)} tensors")
                else:
                    print(f"\t\033[93m{k}:\033[0m {v}")
        print(f'\033[93mlr scheduler state:\033[0m:')
        print("\t\033[93mscheduler type:\033[0m", type(self.lr_scheduler).__name__)
        print("\t\033[93mcurrent lr:\033[0m", self.lr_scheduler.get_last_lr())
        print("\t\033[93mstate_dict:\033[0m", self.lr_scheduler.state_dict())
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n')

        return True
