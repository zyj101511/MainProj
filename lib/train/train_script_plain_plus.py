import os
from torch import nn
from lib.models.mastrack_plain import build_model
from lib.train.trainer.mastrack_trainer import MASTrainer
from lib.train.actor.mastrack_actor_plain import MASTrackActor
from lib.utils.box_ops import giou_loss
from torch.nn.functional import l1_loss
from lib.utils.focal_loss import FocalLoss
from torch.nn import BCEWithLogitsLoss
from lib.train.utils import *
from torch.nn.parallel import DistributedDataParallel as DDP
from lib.config.loader import load_from_yaml


def run(settings):
    if not os.path.exists(settings.cfg_file):
        raise ValueError("%s doesn't exist." % settings.cfg_file)

    cfg = load_from_yaml(settings.cfg_file)

    settings.from_pretrained = cfg.TRAIN.FROM_PRETRAINED
    settings.load_latest_ckpt = cfg.TRAIN.LOAD_LATEST_CKPT

    settings.scheduler_type = cfg.TRAIN.OPTIMIZER.SCHEDULER.TYPE

    settings.save_every_n_epoch = cfg.TRAIN.SAVE_EVERY_N_EPOCH
    settings.save_last_n_ckpt = cfg.TRAIN.SAVE_LAST_N_CKPT
    settings.save_ckpt_list = cfg.TRAIN.SAVE_CKPT_LIST
    settings.p = cfg.MODEL.HEAD.P
    settings.distance_factor = cfg.MODEL.HEAD.DISTANCE_FACTOR
    settings.sample_last_template = cfg.TRAIN.SAMPLE_LAST_TEMPLATE


    if settings.local_rank in [-1, 0]:
        print(f'\033[93mconfiguration is loaded from: \033[0m{settings.cfg_file}\n')

    # update settings based on cfg
    update_settings(settings, cfg)

    # Record the training log
    log_dir = os.path.join(settings.save_dir, 'logs')
    if settings.local_rank in [-1, 0]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    settings.log_file = os.path.join(log_dir, f"{settings.script_name}-{settings.config_name}.log")

    # Build dataloaders
    for name in cfg.DATA.TRAIN.DATASETS_NAME:
        dataset = names2datasets_pp(name, settings, sample_last_template=settings.sample_last_template)
    train_loader = build_train_loader_pp(cfg, dataset, settings)
    loaders = [train_loader]

    # Build network
    net = build_model(cfg, training=True)

    # wrap networks to distributed one
    if settings.local_rank != -1:
        torch.cuda.set_device(settings.local_rank)
        net = net.to(settings.local_rank)
        net = DDP(net, device_ids=[settings.local_rank],
                  output_device=settings.local_rank, find_unused_parameters=True)
        settings.device = torch.device(f"cuda:{settings.local_rank}")

    else:
        net = net.to('cuda:0')
        settings.device = torch.device("cuda:0")


    # Loss function and actor
    focal_loss = FocalLoss()
    objective = {'giou_loss': giou_loss, 'tracking_l1_loss': l1_loss,
                 'focal_loss': focal_loss, 'trajectory_l1_loss': nn.L1Loss(reduction='none'),
                 'cls_loss': BCEWithLogitsLoss()}
    loss_weight = {'tracking': {'giou_loss': cfg.TRAIN.LOSS.GIOU_WEIGHT,
                               'l1_loss': cfg.TRAIN.LOSS.L1_WEIGHT,
                                'focal_loss': cfg.TRAIN.LOSS.FOCAL_WEIGHT,
                                'total': cfg.TRAIN.LOSS.TRACK_WEIGHT},
                   'trajectory': {'decay_factor': cfg.TRAIN.LOSS.DECAY_FACTOR,  # 远未来轨迹loss权重衰减
                                  'near_future_loss': cfg.TRAIN.LOSS.NEAR_FUTURE_WEIGHT,
                                  'distant_future_loss': cfg.TRAIN.LOSS.DISTANT_FUTURE_WEIGHT,
                                  'total': cfg.TRAIN.LOSS.TRAJECTORY_WEIGHT},}
    actor = MASTrackActor(net=net, objective=objective, loss_weight=loss_weight, cfg=cfg)

    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)
    print(f'\033[93moptimizer state:\033[0m:')
    print("\t\033[93moptimizer type:\033[0m", type(optimizer).__name__)
    for i, group in enumerate(optimizer.param_groups):
        print(f"\tgroup {i}:")
        for k, v in group.items():
            if k == "params":
                print(f"\t\033[93mparams:\033[0m {len(v)} tensors")
            else:
                print(f"\t\033[93m{k}:\033[0m {v}")
    print(f'\033[93mlr scheduler state:\033[0m:')
    print("\t\033[93mscheduler type:\033[0m", type(lr_scheduler).__name__)
    print("\t\033[93mcurrent lr:\033[0m", lr_scheduler.get_last_lr())
    print("\t\033[93mstate_dict:\033[0m", lr_scheduler.state_dict())

    use_amp = getattr(cfg.TRAIN, "AMP", False)
    trainer = MASTrainer(actor=actor,
                         loaders=loaders,
                         optimizer=optimizer,
                         lr_scheduler=lr_scheduler,
                         settings=settings,
                         use_amp=use_amp)

    trainer.train(cfg.TRAIN.EPOCH, load_latest=settings.load_latest_ckpt, load_pretrained_ckpt=settings.from_pretrained)
