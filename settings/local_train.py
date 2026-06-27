class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/yanjiezhang/Downloads/Dissertation/MainProj'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/home/yanjiezhang/Downloads/Dissertation/MainProj/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/home/yanjiezhang/Downloads/Dissertation/MainProj/pretrained_networks'
        self.fe108_train_dir = '/home/yanjiezhang/Downloads/Dissertation/dataset/FE108/train'
        self.visevent_train_dir = '/home/yanjiezhang/Downloads/Dissertation/dataset/VisEvent/train'
