import torch
from lib.train.data.loader import MASLoader
from lib.train.data.loader import mas_collate
from lib.utils.misc import is_main_process
from lib.train.data.dataset import FE108Dataset
from lib.train.data.sampler import DistributedTrackingPredSampler


def update_settings(settings, cfg):
    settings.print_interval = cfg.TRAIN.PRINT_INTERVAL
    settings.grad_clip_norm = cfg.TRAIN.GRAD_CLIP_NORM
    settings.print_stats = None
    settings.batchsize = cfg.TRAIN.BATCH_SIZE
    settings.scheduler_type = cfg.TRAIN.OPTIMIZER.SCHEDULER.TYPE
    settings.search_scale_factor = cfg.DATA.SEARCH.SCALE_FACTOR
    settings.search_scale_jitter_factor = cfg.DATA.SEARCH.SCALE_JITTER
    settings.search_output_sz = cfg.DATA.SEARCH.SIZE
    settings.template_output_sz = cfg.DATA.TEMPLATE.SIZE


def names2datasets(name: str, settings):
    assert name in ["FE108", "VISEVENT", 'FELT']
    print("start creating dataset")
    if name == "FE108":
        dataset = FE108Dataset(settings.env.fe108_dir,
                               search_out_sz=settings.search_output_sz,
                               template_out_sz=settings.template_output_sz,
                               scale_factor=settings.search_scale_factor,
                               scale_jitter_factor=settings.search_scale_jitter_factor,
                               split='train')
        print("creating dataset:", name)
    return dataset


def build_train_loader(cfg, dataset, settings):
    print('start building dataloader')

    L_candidates = cfg.TRAIN.L
    P = cfg.MODEL.HEAD.P
    df = cfg.MODEL.HEAD.DISTANCE_FACTOR
    T = cfg.MODEL.T
    samples_per_epoch = cfg.DATA.TRAIN.SAMPLES_PER_EPOCH

    batch_sampler = DistributedTrackingPredSampler(dataset, settings.batchsize, samples_per_epoch,
                                        L_candidates, P=P, distance_factor=df, T=T)

    train_loader = MASLoader(name='train', training=True, dataset=dataset, batch_sampler=batch_sampler,
                       collate_fn=mas_collate, batch_dim=2, num_workers=cfg.TRAIN.NUM_WORKERS, epoch_interval=1)
    return train_loader

def build_test_loader(cfg, dataset, settings):
    raise NotImplementedError("Build test loader is not implemented yet.")

def get_optimizer_scheduler(net, cfg):
    param_dicts = [
        {"params": [p for n, p in net.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in net.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": cfg.TRAIN.OPTIMIZER.LR * cfg.TRAIN.OPTIMIZER.BACKBONE_MULTIPLIER,
        },
    ]
    if is_main_process():
        print("\033[93mLearnable parameters are shown below.\033[0m")
        for n, p in net.named_parameters():
            if p.requires_grad:
                print(n)

    if cfg.TRAIN.OPTIMIZER.TYPE == "ADAMW":
        optimizer = torch.optim.AdamW(param_dicts, lr=cfg.TRAIN.OPTIMIZER.LR,
                                      weight_decay=cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY)
    else:
        raise ValueError("Unsupported Optimizer")
    if cfg.TRAIN.OPTIMIZER.SCHEDULER.TYPE == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.TRAIN.OPTIMIZER.SCHEDULER.LR_DROP_EPOCH)
    else:
        raise ValueError("Unsupported scheduler")
    return optimizer, lr_scheduler


if __name__ == '__main__':
    from lib.config.loader import load_from_yaml
    cfg = load_from_yaml('/home/yanjiezhang/Downloads/Dissertation/MainProj/experiments/fe108_mastrack.yaml')
    from easydict import EasyDict
    settings = EasyDict()
    settings.env = EasyDict()
    update_settings(settings, cfg)
    settings.env.fe108_dir = '/home/yanjiezhang/Downloads/Dissertation/dataset/FE108_nbinsGTP_lmdb'
    dataset = names2datasets("FE108", settings)
    train_loader = build_train_loader(cfg, dataset, settings)
    batch = next(iter(train_loader))
    print(batch.keys())
    print(len(batch))
    print(batch['search'].shape)  # (B, L, T, C, H, W)
    print(batch['search_anno'].shape)  # (B, L+df*P, 4)
    print(batch['template'].shape)  # # (B, L, T, C, H, W)

    net = torch.nn.Linear(10, 1)  # Example model
    opt, sch = get_optimizer_scheduler(net, cfg)

