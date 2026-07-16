from dataclasses import dataclass, field


@dataclass
class _Dataset:
    DATASETS_NAME: list[str]  # 多个数据集的名字
    DATASETS_RATIO: list[int]  # 多个数据集的采样比例
    SAMPLES_PER_EPOCH: int  # 每个epoch采样多少样本
    def __post_init__(self):
        if len(self.DATASETS_NAME) != len(self.DATASETS_RATIO):
            raise ValueError("DATASETS_NAME and DATASETS_RATIO must have the same length")
        if not self.DATASETS_NAME:
            raise ValueError("DATASETS_NAME must not be empty")
        if any(not isinstance(r, int) or r <= 0 for r in self.DATASETS_RATIO):
            raise ValueError("DATASETS_RATIO values must be positive integer")

@dataclass
class _DataCrop:
    SIZE:int   # 输入给模型的高宽
    SCALE_FACTOR: float
    SCALE_JITTER: float
    CTR_JITTER: float

@dataclass
class _DATA:
    # TRAIN
    TRAIN: _Dataset = field(default_factory=lambda: _Dataset(
        DATASETS_NAME=["FE108"],
        DATASETS_RATIO=[1],
        SAMPLES_PER_EPOCH=60000,
    ))
    # VAL
    VAL: _Dataset = field(default_factory=lambda: _Dataset(
        DATASETS_NAME=["FE108"],
        DATASETS_RATIO=[1],
        SAMPLES_PER_EPOCH=1
    ))
    # SEARCH
    SEARCH: _DataCrop = field(default_factory=lambda: _DataCrop(
        SIZE=256,
        SCALE_FACTOR=1.2,
        SCALE_JITTER=0.5,
        CTR_JITTER=0.2
    ))
    # TEMPLATE
    TEMPLATE: _DataCrop = field(default_factory=lambda: _DataCrop(
        SIZE=128,
        SCALE_FACTOR=1,
        SCALE_JITTER=1,
        CTR_JITTER=0.2
    ))

@dataclass
class _Backbone:
    TYPE: str
    STRIDE: int  # Backbone下采样倍数

@dataclass
class _Head:
    TYPE: str  # 输出头类型: CENTER, CORNER
    NUM_CHANNELS: int  # 中间层通道数
    P: int  # 轨迹头的预测距离, P是短距离
    DISTANCE_FACTOR: int  # 距离因子, P*DISTANCE_FACTOR是预测的长距离

@dataclass
class _Multi_Timescale_Memory:
    NUM_BRANCHES: int  # 多时间尺度分支数
    NUM_LAYERS: int  # 每个分支的point-wise conv层数

@dataclass
class _MODEL:
    NEURON: str = 'MLIF'
    T: int = 1 # 神经元的输入步数
    BACKBONE: _Backbone = field(default_factory=lambda: _Backbone(
        TYPE = 'LARGE',
        STRIDE = 16
    ))
    MULTI_TIMESCALE_MODULE: _Multi_Timescale_Memory = field(default_factory=lambda: _Multi_Timescale_Memory(
        NUM_BRANCHES = 4,
        NUM_LAYERS = 1
    ))
    HEAD: _Head = field(default_factory=lambda: _Head(
        TYPE = 'CENTER',
        NUM_CHANNELS = 256,
        P = 4,
        DISTANCE_FACTOR = 4
    ))
    def __post_init__(self):
        if self.NEURON == 'MILIF' and (not isinstance(self.D, int) or self.D <= 0):
            raise ValueError("D has to be set to a positive integer when using MILIF neuron")

@dataclass
class _SCHEDULER:
    TYPE: str = 'step'
    DECAY_RATE: float = 0.1
    LR_DROP_EPOCH: int = 200

@dataclass
class _OPTIMIZER:
    TYPE: str = 'ADAMW'
    LR: float = 0.0004
    # 学习率调整策略,step就是多少个epoch调整,
    SCHEDULER: _SCHEDULER = field(default_factory=_SCHEDULER)
    WEIGHT_DECAY: float = 0.0001
    BACKBONE_MULTIPLIER: float = 1  # backbone一般是预训练好的,用更小的学习率,其他新加层用LR

@dataclass
class _LOSS:
    FOCAL_WEIGHT: float = 1.0
    GIOU_WEIGHT: float = 2.0
    L1_WEIGHT: float = 4.0
    DECAY_FACTOR: float = 0.9
    NEAR_FUTURE_WEIGHT: float = 10
    DISTANT_FUTURE_WEIGHT: float = 0.05
    TRACK_WEIGHT: float = 1.0
    TRAJECTORY_WEIGHT: float = 3.0

@dataclass
class _TRAIN:
    EPOCH: int = 500
    BATCH_SIZE: int = 2
    L: list = field(default_factory=lambda: [1, 2, 4])  # 训练时, 采样序列的长度, 训练时会随机选择一个长度
    NUM_WORKERS: int = 0  # Dataloader用多少个子进程
    OPTIMIZER: _OPTIMIZER = field(default_factory=_OPTIMIZER)
    LOSS: _LOSS = field(default_factory=_LOSS)
    PRINT_INTERVAL: int = 50  # 多少个batch打印一次训练状态
    VAL_EPOCH_INTERVAL: int | None = None  # 多少个epoch跑一次验证集, None不开启VAL
    GRAD_CLIP_NORM: float = 0.1  # 梯度范数太大时梯度裁剪的阈值
    AMP: bool = False  # 是否启用混合精度
    FROM_PRETRAINED: bool = False  # 是否从预训练模型开始训练, 如果为False, 则从头开始训练
    LOAD_LATEST_CKPT: bool = False  # 是否加载最近的ckpt继续训练, 如果为False, 则从头开始训练
    PRETRAINED_FILE_NAME: str | None = None  # 指定预训练模型的文件名
    SAVE_EVERY_N_EPOCH: int = 5  # 每多少个epoch保存一次模型
    SAVE_LAST_N_CKPT: int = 5  # 保存最近多少个ckpt, 其他的会被删除, 如果为None, 则不删除
    SAVE_CKPT_LIST: list[str] = field(default_factory=list)  # 指定保存的ckpt文件名, 如果为空, 则保存所有ckpt
    SAMPLE_LAST_TEMPLATE: float = 0.5  # 训练时, 采样模板帧的策略, 0.5表示50%的概率采样上一帧, 50%的概率采样之前的任意一帧

@dataclass
class _TEST:
    DEBUG: int = 0
    SAVE_PLOT: bool = False
    PRETRAINED_FILE_NAME: str | None = None
    SEARCH_SCALE_FACTOR: float = 5
    TEMPLATE_SCALE_FACTOR: float = 1.2
    CLIP_BOX_MARGIN: int = 10
    SEARCH_BOX_UPDATE_MARGIN_RATIO: float = 0.1

@dataclass
class Config:
    DATA: _DATA = field(default_factory=_DATA)
    MODEL: _MODEL = field(default_factory=_MODEL)
    TRAIN: _TRAIN = field(default_factory=_TRAIN)
    TEST: _TEST = field(default_factory=_TEST)




