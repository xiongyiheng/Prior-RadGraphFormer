import os
from monai.engines import SupervisedTrainer
from monai.inferers import SimpleInferer
from monai.handlers import LrScheduleHandler, ValidationHandler, StatsHandler, TensorBoardStatsHandler, CheckpointSaver, \
    MeanDice
import torch
import gc
from utils import get_total_grad_norm


# define customized trainer
class RelationformerTrainer(SupervisedTrainer):

    def __init__(self, writer, **kwargs):
        self.distributed = kwargs.pop('distributed')
        # Initialize superclass things
        super().__init__(**kwargs)
        self.tl = ['tokens', 'labels', 'edges']
        self.writer = writer

    def _iteration(self, engine, batchdata):
        images = [image.to(engine.state.device, non_blocking=False) for image in batchdata['imgs_ls']]
        target = []
        for i in range(len(images)):  # iterate batch
            current_target = {'tokens': batchdata['tokens'][i].to(engine.state.device, non_blocking=True),
                              'labels': batchdata['labels'][i].to(engine.state.device, non_blocking=True),
                              'edges': batchdata['edges'][i].to(engine.state.device, non_blocking=True)}
            target.append(current_target)

        self.network.train()
        self.optimizer.zero_grad()
        h, out = self.network(images)

        losses = self.loss_function(h, out, target)
        self.writer.add_scalar("train_token_classification_loss", losses['tokens'].item(), engine.state.iteration)
        self.writer.add_scalar("train_label_classification_loss", losses['labels'].item(), engine.state.iteration)
        self.writer.add_scalar("train_edge_loss", losses['edges'].item(), engine.state.iteration)
        self.writer.add_scalar("train_total_loss", losses['total'].item(), engine.state.iteration)
        # Clip the gradient
        # clip_grad_norm_(
        #     self.network.parameters(),
        #     max_norm=GRADIENT_CLIP_L2_NORM,
        #     norm_type=2,
        # )
        losses['total'].backward()

        if 0.1 > 0:  # todo replace
            _ = torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.1)
        else:
            _ = get_total_grad_norm(self.networm.parameters(), 0.1)

        self.optimizer.step()

        gc.collect()
        torch.cuda.empty_cache()
        return {"images": images, "loss": losses}


def build_trainer(train_loader, net, loss, optimizer, scheduler, writer,
                  evaluator, config, device, fp16=False, distributed=False, local_rank=0):
    """[summary]

    Args:
        train_loader ([type]): [description]
        net ([type]): [description]
        loss ([type]): [description]
        optimizer ([type]): [description]
        evaluator ([type]): [description]
        scheduler ([type]): [description]
        max_epochs ([type]): [description]

    Returns:
        [type]: [description]
    """
    train_handlers = [
        LrScheduleHandler(
            lr_scheduler=scheduler,
            print_lr=True,
            epoch_level=True,
        ),
        ValidationHandler(
            validator=evaluator,
            interval=config.TRAIN.VAL_INTERVAL,
            epoch_level=True
        )
    ]
    if local_rank == 0:
        train_handlers.extend(
            [
                CheckpointSaver(
                    save_dir=os.path.join(config.TRAIN.SAVE_PATH, "runs",
                                          '%s_%d' % (config.log.exp_name, config.DATA.SEED), 'models'),
                    save_dict={"net": net, "optimizer": optimizer, "scheduler": scheduler},
                    save_interval=1,
                    n_saved=1
                ),
            ]
        )

    trainer = RelationformerTrainer(
        device=device,
        max_epochs=config.TRAIN.EPOCHS,
        train_data_loader=train_loader,
        network=net,
        optimizer=optimizer,
        loss_function=loss,
        inferer=SimpleInferer(),
        train_handlers=train_handlers,
        distributed=distributed,
        writer=writer
    )

    return trainer
