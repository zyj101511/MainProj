class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/user/work/rm25043/Dissertation/MainProj'  # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/user/work/rm25043/Dissertation/MainProj/tensorboard'    # Directory for tensorboard files.
        self.pretrained_ckpt_dir = None
        self.fe108_dir = '/user/work/rm25043/Dissertation/dataset/FE108_nbinsGTP_lmdb'
        self.visevent_dir = '/user/work/rm25043/Dissertation/dataset/VisEvent/train'
