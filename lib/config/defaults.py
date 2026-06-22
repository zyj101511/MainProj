from dataclasses import dataclass, field


@dataclass
class _Dataset:
    DATASETS_NAME: list[str]
    DATASETS_RATIO: list[int]  # 多个数据集的采样比例
    SAMPLE_PER_EPOCH: int  # 每个epoch采样多少样本
    def __post_init__(self):
        if len(self.DATASETS_NAME) != len(self.DATASETS_RATIO):
            raise ValueError("DATASETS_NAME and DATASETS_RATIO must have the same length")
        if not self.DATASETS_NAME:
            raise ValueError("DATASETS_NAME must not be empty")
        if any(not isinstance(r, int) or r <= 0 for r in self.DATASETS_RATIO):
            raise ValueError("DATASETS_RATIO values must be positive integer")

@dataclass
class _DataCrop:
    HEIGHT: int  # 输入给模型的高宽
    WIDTH: int
    FACTOR: float # 裁切大小对于BBOX的缩放倍数
    CENTER_JITTER: float
    SCALE_JITTER: float
    NUMBER: int

@dataclass
class _DATA:
    SAMPLER_MODE: str = "causal"  # SDTrack采样方法看/draft/todo.md
    MAX_SAMPLE_INTERVAL: int = 200  # 控制采样时template和search两帧的最大间隔
    MEAN: list[float] = field(default_factory=lambda: [0.485, 0.456, 0.406]) # 预训练数据集mean和std, 用于图像归一化
    STD: list[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    # TRAIN
    TRAIN: _Dataset = field(default_factory=lambda: _Dataset(
        DATASETS_NAME=["FE108", "VisEvent"],
        DATASETS_RATIO=[1, 1],
        SAMPLE_PER_EPOCH=60000
    ))
    # VAL
    VAL: _Dataset = field(default_factory=lambda: _Dataset(
        DATASETS_NAME=["FE108"],
        DATASETS_RATIO=[1],
        SAMPLE_PER_EPOCH=5000
    ))
    # SEARCH
    SEARCH: _DataCrop = field(default_factory=lambda: _DataCrop(
        HEIGHT=256,
        WIDTH=256,
        FACTOR=5.0,
        CENTER_JITTER=4.5,
        SCALE_JITTER=0.5,
        NUMBER=1,
    ))
    # TEMPLATE
    TEMPLATE: _DataCrop = field(default_factory=lambda: _DataCrop(
        HEIGHT=128,
        WIDTH=128,
        FACTOR=2.0,
        CENTER_JITTER=0.0,
        SCALE_JITTER=0.0,
        NUMBER=1,
    ))

@dataclass
class _Backbone:
    TYPE: str
    STRIDE: int  # Backbone下采样倍数

@dataclass
class _Head:
    TYPE: str  # 输出头类型: CENTER, CORNER
    NUM_CHANNELS: int  # 中间层通道数

@dataclass
class _MODEL:
    NEURON: str = 'LIF'
    T: int = 1 # 神经元的输入步数
    D: int | None = None# I-LIF的内部状态步数
    PRETRAINED_FILE: str | None = None
    BACKBONE: _Backbone = field(default_factory=lambda: _Backbone(
        TYPE = 'BASE',
        STRIDE = 16
    ))
    HEAD: _Head = field(default_factory=lambda: _Head(
        TYPE = 'CENTER',
        NUM_CHANNELS = 256
    ))
    def __post_init__(self):
        if self.NEURON == 'I-LIF' and (not isinstance(self.D, int) or self.D <= 0):
            raise ValueError("D has to be set to a positive integer when using I-LIF neuron")

@dataclass
class _SCHEDULER:
    TYPE: str = 'step'
    DECAY_RATE: float = 0.1
    LR_DROP_EPOCH: int = 100

@dataclass
class _OPTIMIZER:
    TYPE: str = 'ADAMW'
    LR: float = 0.0004
    # 学习率调整策略,step就是多少个epoch调整,
    SCHEDULER: _SCHEDULER = field(default_factory=_SCHEDULER)
    WEIGHT_DECAY: float = 0.0001
    BACKBONE_MULTIPLIER: float = 0.1  # backbone一般是预训练好的,用更小的学习率,其他新加层用LR

@dataclass
class _LOSS:
    FOCAL_WEIGHT: float = 1.0
    GIOU_WEIGHT: float = 2.0
    L1_WEIGHT: float = 5.0

@dataclass
class _TRAIN:
    EPOCH: int = 100
    BATCH_SIZE: int = 8
    NUM_WORKERS: int = 0  # Dataloader用多少个子进程
    OPTIMIZER: _OPTIMIZER = field(default_factory=_OPTIMIZER)
    LOSS: _LOSS = field(default_factory=_LOSS)
    PRINT_INTERVAL: int = 50  # 多少个batch打印一次训练状态
    VAL_EPOCH_INTERVAL: int | None = None  # 多少个epoch跑一次验证集, None不开启VAL
    GRAD_CLIP_NORM: float = 0.1  # 梯度范数太大时梯度裁剪的阈值
    AMP: bool = False  # 是否启用混合精度

@dataclass
class _TEST:
    CHECKPOINT_EPOCH: int = 100
    CHECKPOINT_PATH: str | None = None

@dataclass
class Config:
    DATA: _DATA = field(default_factory=_DATA)
    MODEL: _MODEL = field(default_factory=_MODEL)
    TRAIN: _TRAIN = field(default_factory=_TRAIN)
    TEST: _TEST = field(default_factory=_TEST)




