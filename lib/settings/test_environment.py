import importlib
import os


class EnvSettings_ITP:
    def __init__(self, workspace_dir='', data_dir='', save_dir=''):
        self.workspace_dir = workspace_dir
        self.save_dir = save_dir
        self.results_txt_path = os.path.join(save_dir, 'txt_results')
        self.result_plot_path = os.path.join(save_dir, 'plot_results')
        self.checkpoint_path = ''
        self.fe108_dir = os.path.join(data_dir, 'FE108_GTP_lmdb')
        self.visevent_dir = os.path.join(data_dir, 'VisEvent/test')


def create_default_local_file_ITP_test(workspace_dir, data_dir, save_dir):
    comment = {'results_path': 'Where to store tracking results',
               'network_path': 'Where tracking networks are stored.'}

    path = os.path.join(os.path.dirname(__file__), 'local_test_local.py')
    with open(path, 'w') as f:
        settings = EnvSettings_ITP(workspace_dir, data_dir, save_dir)

        f.write('from settings.test_environment import EnvSettings_ITP\n\n')
        f.write('def local_test_env_settings():\n')
        f.write('    settings = EnvSettings_ITP()\n\n')
        f.write('    # Set your local paths here.\n\n')

        for attr, attr_val in settings.__dict__.items():
            comment_str = None
            if attr in comment:
                comment_str = comment[attr]

            if comment_str is None:
                f.write('    settings.{} = \'{}\'\n'.format(attr, attr_val))
            else:
                f.write('    settings.{} = \'{}\'    # {}\n'.format(attr, attr_val, comment_str))

        f.write('\n    return settings\n\n')


def env_settings():
    env_module_name = 'lib.settings.local_test'
    try:
        env_module = importlib.import_module(env_module_name)
        return env_module.local_test_env_settings()
    except:
        env_file = os.path.join(os.path.dirname(__file__), 'local_test_local.py')
        raise RuntimeError('YOU HAVE NOT SETUP YOUR local.py!!!\n Go to "{}" and set all the paths you need. '
                           'Then try to run again.'.format(env_file))
if __name__ == '__main__':
    create_default_local_file_ITP_test('/home/yanjiezhang/Downloads/Dissertation/MainProj',
                                   '/home/yanjiezhang/Downloads/Dissertation/dataset',
                                   '/home/yanjiezhang/Downloads/Dissertation/MainProj/tracking_results')