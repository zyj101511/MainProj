import importlib
import os
from collections import OrderedDict


def create_default_local_file_ITP_train(workspace_dir, data_dir):
    path = os.path.join(os.path.dirname(__file__), 'local_train.py')

    empty_str = '\'\''
    default_settings = OrderedDict({
        'workspace_dir': workspace_dir,
        'tensorboard_dir': os.path.join(workspace_dir, 'tensorboard'),    # Directory for tensorboard files.
        'pretrained_networks': os.path.join(workspace_dir, 'pretrained_networks'),
        'fe108_train_dir': os.path.join(data_dir, 'FE108/train'),
        'visevent_train_dir': os.path.join(data_dir, 'VisEvent/train')})

    comment = {'workspace_dir': 'Base directory for saving network checkpoints.',
               'tensorboard_dir': 'Directory for tensorboard files.'}

    with open(path, 'w') as f:
        f.write('class EnvironmentSettings:\n')
        f.write('    def __init__(self):\n')

        for attr, attr_val in default_settings.items():
            comment_str = None
            if attr in comment:
                comment_str = comment[attr]
            if comment_str is None:
                if attr_val == empty_str:
                    f.write('        self.{} = {}\n'.format(attr, attr_val))
                else:
                    f.write('        self.{} = \'{}\'\n'.format(attr, attr_val))
            else:
                f.write('        self.{} = \'{}\'    # {}\n'.format(attr, attr_val, comment_str))


def env_settings(workspace_dir, data_dir):
    env_module_name = 'settings.local_train'
    try:
        env_module = importlib.import_module(env_module_name)
        return env_module.EnvironmentSettings()
    except:
        env_file = os.path.join(os.path.dirname(__file__), 'local_train.py')

        create_default_local_file_ITP_train(workspace_dir, data_dir)
        raise RuntimeError('YOU HAVE NOT SETUP YOUR local.py!!!\n Go to "{}" and set all the paths you need. Then try to run again.'.format(env_file))
