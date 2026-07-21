class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/yanjiezhang/Downloads/Dissertation/MainProj'  # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/home/yanjiezhang/Downloads/Dissertation/MainProj/tensorboard'    # Directory for tensorboard files.
        self.pretrained_ckpt_dir = None
        self.fe108_dir = '/home/yanjiezhang/Downloads/Dissertation/dataset/FE108_GTP_lmdb'
        self.visevent_dir = '/home/yanjiezhang/Downloads/Dissertation/dataset/VisEvent/train'
