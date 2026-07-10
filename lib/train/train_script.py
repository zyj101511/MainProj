import os
from torch import nn
from lib.models.mastrack import build_model
from lib.train.trainer.mastrack_trainer import MASTrainer
from lib.train.actor.mastrack_actor import MASTrackActor
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

    settings.scheduler_type = cfg.TRAIN.OPTIMIZER.SCHEDULER.TYPE

    # save ckpt if condition is met
    settings.save_every_n_epoch = cfg.TRAIN.SAVE_EVERY_N_EPOCH
    settings.save_last_n_ckpt = cfg.TRAIN.SAVE_LAST_N_CKPT
    settings.save_ckpt_list = cfg.TRAIN.SAVE_CKPT_LIST
    settings.p = cfg.MODEL.HEAD.P
    settings.distance_factor = cfg.MODEL.HEAD.DISTANCE_FACTOR


    if settings.local_rank in [-1, 0]:
        print(f'configuration is loaded from {settings.cfg_file}')

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
        dataset = names2datasets(name, settings)
    train_loader = build_train_loader(cfg, dataset, settings)
    loaders = [train_loader]

    # Build network
    if cfg.TRAIN.PRETRAINED_FILE_NAME is not None:
        net = build_model(cfg, training=True)
    else:
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
    use_amp = getattr(cfg.TRAIN, "AMP", False)
    trainer = MASTrainer(actor=actor,
                         loaders=loaders,
                         optimizer=optimizer,
                         lr_scheduler=lr_scheduler,
                         settings=settings,
                         use_amp=use_amp)

    trainer.train(cfg.TRAIN.EPOCH, load_latest=False, load_pretrained_ckpt=False)
