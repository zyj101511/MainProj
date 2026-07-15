from lib.settings.train_environment import env_settings as train_env_settings
from lib.settings.test_environment import env_settings as test_env_settings


class Settings:
    """ Training settings, e.g. the paths to datasets and networks."""
    def __init__(self, training=True):
        self.set_default(training)

    def set_default(self, training=True):
        if training:
            self.env = train_env_settings()
        else:
            self.env = test_env_settings()
        self.use_gpu = True


