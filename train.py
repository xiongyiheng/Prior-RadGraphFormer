from argparse import ArgumentParser
import os
import yaml
import json
from shutil import copyfile
import torch
from torch.utils.tensorboard import SummaryWriter

from datasets.radgraph.radgraph import Radgraph

from trainer import build_trainer
from models.relationformer_2D import build_relationformer as build_model
from models.matcher_scene import build_matcher
from losses import SetCriterion

import ignite.distributed as igdist

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
# /home/guests/mlmi_kamilia/Radgraphformer_prior/trained_weights/asm=0/runs/RadGraph Experiment_10/models/checkpoint_epoch=3.pt
def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--config',
                        default='./configs/radgraph.yaml',
                        help='config file (.yml) containing the hyper-parameters for training. '
                             'If None, use the nnU-Net config. See /config for examples.')
    parser.add_argument('--resume', default='/home/guests/mlmi_kamilia/Radgraphformer_prior/trained_weights/asm=0/runs/RadGraph Experiment_10/models/checkpoint_epoch=4.pt', type=str, help='checkpoint of the last epoch of the model')
    parser.add_argument('--device', default='cuda', help='device to use for training')
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--nproc_per_node", default=None, type=int)
    parser.add_argument('--cuda_visible_device', nargs='*', type=int, default=None,
                        help='list of index where skip conn will be made')
    parser.add_argument('-batch_size', dest='batch_size', help='batch size', type=int, default=32)
    return parser.parse_args()


class obj:
    def __init__(self, dict1):
        self.__dict__.update(dict1)


def dict2obj(dict1):
    return json.loads(json.dumps(dict1), object_hook=obj)


def match_name_keywords(n, name_keywords):
    out = False
    for b in name_keywords:
        if b in n:
            out = True
            break
    return out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(args):
    # Load the config files
    with open(args.config) as f:
        print('\n*** Config file')
        print(args.config)
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = dict2obj(config)
    config.MODEL.RESUME = args.resume
    config.DATA.BATCH_SIZE = args.batch_size

    print('Experiment Name : ', config.log.exp_name)
    print('Batch size : ', config.DATA.BATCH_SIZE)
    exp_path = os.path.join(config.TRAIN.SAVE_PATH, "runs", '%s_%d' % (config.log.exp_name, config.DATA.SEED))
    if os.path.exists(exp_path) and args.resume == None:
        print('WARNING: Experiment folder exist, please change exp name in config file')
        pass  # TODO: ask for overwrite permission
    elif not len(config.MODEL.RESUME) > 0:
        os.makedirs(exp_path, exist_ok=True)
        copyfile(args.config, os.path.join(exp_path, "config.yaml"))

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.multiprocessing.set_sharing_strategy('file_system')

    train_ds = Radgraph(is_train=True, is_augment=False)
    val_ds = Radgraph(is_train=False, is_augment=False)

    if igdist.get_local_rank() == 0:
        # Ensure that only local rank 0 download the dataset
        igdist.barrier()

    train_loader = igdist.auto_dataloader(train_ds,
                                          batch_size=config.DATA.BATCH_SIZE,
                                          num_workers=config.DATA.NUM_WORKERS,
                                          pin_memory=True,
                                          shuffle=True)
    val_loader = igdist.auto_dataloader(val_ds,
                                        batch_size=config.DATA.BATCH_SIZE,
                                        num_workers=config.DATA.NUM_WORKERS,
                                        pin_memory=True,
                                        shuffle=False)

    device = torch.device(args.device)

    # BUILD MODEL
    model = build_model(config)
    print('Number of parameters : ', count_parameters(model))
    # if config.MODEL.DECODER.FREQ_BIAS: # use freq bias
    #     logsoftmax = True if hasattr(config.MODEL.DECODER,'LOGSOFTMAX_FREQ') and config.MODEL.DECODER.LOGSOFTMAX_FREQ else False
    #     freq_baseline = FrequencyBias(config.DATA.FREQ_BIAS, train_ds, dropout=config.MODEL.DECODER.FREQ_DR, logsoftmax=logsoftmax)

    net_wo_dist = model.to(device)
    # freq_baseline = freq_baseline.to(device) if config.MODEL.DECODER.FREQ_BIAS else None

    model = igdist.auto_model(model)
    # freq_baseline = igdist.auto_model(freq_baseline) if config.MODEL.DECODER.FREQ_BIAS and logsoftmax else None
    freq_baseline = None

    matcher = build_matcher(config=config)
    asm = model.asm.to(device)
    asm = igdist.auto_model(asm)
    project = model.project.to(device)
    project = igdist.auto_model(project)
    loss = SetCriterion(config, matcher, asm, project=project,
                        freq_baseline=freq_baseline if config.MODEL.DECODER.FREQ_BIAS else None,
                        use_target=True, focal_alpha=config.TRAIN.FOCAL_LOSS_ALPHA).to(
        device)  # use target uses gt label for freq baseline

    param_dicts = [
        {
            "params":
                [p for n, p in model.named_parameters()
                 if
                 not match_name_keywords(n, ["backbone", 'reference_points', 'sampling_offsets']) and p.requires_grad],
            "lr": float(config.TRAIN.LR)
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       match_name_keywords(n, ["backbone"]) and p.requires_grad],
            "lr": float(config.TRAIN.LR_BACKBONE)
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       match_name_keywords(n, ['reference_points', 'sampling_offsets']) and p.requires_grad],
            "lr": float(config.TRAIN.LR) * 0.1
        }

    ]

    optimizer = torch.optim.AdamW(
        param_dicts, lr=float(config.TRAIN.LR), weight_decay=float(config.TRAIN.WEIGHT_DECAY)
    )
    optimizer = igdist.auto_optim(optimizer)

    # LR schedular
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.TRAIN.LR_DROP)

    if len(config.MODEL.RESUME) > 1 or len(config.MODEL.PRETRAIN) > 1:
        assert not (len(config.MODEL.RESUME) > 0 and len(
            config.MODEL.PRETRAIN) > 0), 'Both pretrain and resume cant be used together'
        ckpt_path = config.MODEL.RESUME if len(config.MODEL.RESUME) > 0 else config.MODEL.PRETRAIN
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        missing_keys, unexpected_keys = net_wo_dist.load_state_dict(checkpoint['net'], strict=False)
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        if len(config.MODEL.RESUME) > 0:
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            last_epoch = scheduler.last_epoch

    writer = SummaryWriter(
        log_dir=os.path.join(config.TRAIN.SAVE_PATH, "runs", '%s_%d' % (config.log.exp_name, config.DATA.SEED)),
    )
    from evaluator import build_evaluator

    evaluator = build_evaluator(
        val_loader,
        model,
        optimizer,
        scheduler,
        writer,
        config,
        device,
        loss
    )

    trainer = build_trainer(
        train_loader,
        model,
        loss,
        optimizer,
        scheduler,
        writer,
        evaluator,
        config,
        device
    )

    if len(config.MODEL.RESUME) > 0:
        # evaluator.state.epoch = last_epoch
        trainer.state.epoch = last_epoch
        trainer.state.iteration = trainer.state.epoch_length * last_epoch

    # logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    # if args.eval:
    #     evaluator.run()
    # else:
    trainer.run()


if __name__ == '__main__':
    args = parse_args()
    main(args)
