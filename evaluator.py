import os
import gc
import torch
import torch.nn.functional as F
import numpy as np
import time

from monai.engines import SupervisedEvaluator
from monai.handlers import StatsHandler, CheckpointSaver, TensorBoardStatsHandler

from argparse import ArgumentParser

from inference import graph_infer
from torch.utils.data import DataLoader
from typing import TYPE_CHECKING, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from monai.config import IgniteInfo
from monai.engines.utils import default_metric_cmp_fn, default_prepare_batch
from monai.inferers import Inferer, SimpleInferer
from monai.transforms import Transform
from monai.utils import ForwardMode, min_version, optional_import
from ignite.engine import Events

from util.radgraph_eval import BasicRadGraphEvaluator

if TYPE_CHECKING:
    from ignite.engine import Engine, EventEnum
    from ignite.metrics import Metric
else:
    Engine, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Engine")
    Metric, _ = optional_import("ignite.metrics", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Metric")
    EventEnum, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "EventEnum")

parser = ArgumentParser()
parser.add_argument('--config',
                    default='./configs/radgraph.yaml',
                    help='config file (.yml) containing the hyper-parameters for training. '
                         'If None, use the nnU-Net config. See /config for examples.')
parser.add_argument('--resume', default=None, help='checkpoint of the last epoch of the model')
parser.add_argument('--device', default='cuda',
                    help='device to use for training')
parser.add_argument('--cuda_visible_device', nargs='*', type=int, default=None,
                    help='list of index where skip conn will be made')


# Define customized evaluator
class RelationformerEvaluator(SupervisedEvaluator):
    def __init__(
            self,
            device: torch.device,
            val_data_loader: Union[Iterable, DataLoader],
            network: torch.nn.Module,
            epoch_length: Optional[int] = None,
            non_blocking: bool = False,
            prepare_batch: Callable = default_prepare_batch,
            iteration_update: Optional[Callable] = None,
            inferer: Optional[Inferer] = None,
            postprocessing: Optional[Transform] = None,
            key_val_metric: Optional[Dict[str, Metric]] = None,
            additional_metrics: Optional[Dict[str, Metric]] = None,
            metric_cmp_fn: Callable = default_metric_cmp_fn,
            val_handlers: Optional[Sequence] = None,
            amp: bool = False,
            mode: Union[ForwardMode, str] = ForwardMode.EVAL,
            event_names: Optional[List[Union[str, EventEnum]]] = None,
            event_to_attr: Optional[dict] = None,
            decollate: bool = True,
            **kwargs,
    ) -> None:
        super().__init__(
            device=device,
            val_data_loader=val_data_loader,
            epoch_length=epoch_length,
            non_blocking=non_blocking,
            prepare_batch=prepare_batch,
            iteration_update=iteration_update,
            postprocessing=postprocessing,
            key_val_metric=key_val_metric,
            additional_metrics=additional_metrics,
            metric_cmp_fn=metric_cmp_fn,
            val_handlers=val_handlers,
            amp=amp,
            mode=mode,
            event_names=event_names,
            event_to_attr=event_to_attr,
            decollate=decollate,
            network=network,
            inferer=SimpleInferer() if inferer is None else inferer,
        )

        self.config = kwargs.pop('config')
        self.rg_evaluator = BasicRadGraphEvaluator()
        self.freq_baseline = None
        if 'freq_baseline' in kwargs.keys():
            self.freq_baseline = kwargs['freq_baseline']
        self.writer = kwargs.pop('writer')
        self.loss_function = kwargs.pop('loss_function')
        self._accumulate()
        self.add_emd_rel = self.config.MODEL.DECODER.ADD_EMB_REL

    def _iteration(self, engine, batchdata):
        start = time.time()
        images = [image.to(engine.state.device, non_blocking=False) for image in batchdata['imgs_ls']]
        gt_datas = []
        tokens = []
        labels = []
        edges = []
        target = []  # for loss
        for i in range(batchdata['imgs_ls'].shape[0]):  # iterate batch
            current_target = {}
            current_target['tokens'] = batchdata['tokens'][i]
            current_target['labels'] = batchdata['labels'][i]
            current_target['edges'] = batchdata['edges'][i]
            gt_datas.append(current_target)

            tokens.append(batchdata['tokens'][i].cpu().numpy())
            labels.append(batchdata['labels'][i].cpu().numpy())
            edges.append(batchdata['edges'][i].cpu().numpy())

            current_target_for_loss = {}
            current_target_for_loss['tokens'] = batchdata['tokens'][i].to(engine.state.device, non_blocking=True)
            current_target_for_loss['labels'] = batchdata['labels'][i].to(engine.state.device, non_blocking=True)
            current_target_for_loss['edges'] = batchdata['edges'][i].to(engine.state.device, non_blocking=True)
            target.append(current_target_for_loss)
        self.network.eval()
        h, out = self.network(images)  # todo output logit and edge are same value

        losses = self.loss_function(h, out, target)

        asm = self.network.asm
        project = self.network.project
        relation_embed = self.network.relation_embed
        out = graph_infer(h, out, relation_embed=relation_embed, asm=asm, project=project, freq=self.freq_baseline, emb=self.add_emd_rel)

        pred_edges = [{'pred_rels': pred_rels, 'pred_edge': pred_edge, 'pred_rel_score': pred_rel_score} for
                      pred_rels, pred_edge, pred_rel_score in
                      zip(out['pred_rels'], out['pred_rels_class'], out['pred_rels_score'])]
        pred_nodes = [{'tokens': pred_token, 'labels': pred_label} for pred_token, pred_label in
                      zip(out['pred_tokens'], out['pred_labels'])]

        for i, (gt_data, pred_node, pred_edge) in enumerate(zip(gt_datas, pred_nodes, pred_edges)):
            self.rg_evaluator.evaluate_radgraph_entry(gt_data, [pred_node, pred_edge], losses)

        gc.collect()
        torch.cuda.empty_cache()

        return {**{"images": images, "tokens": tokens, "labels": labels, "edges": edges}, **out}

    def _accumulate(self):

        @self.on(Events.EPOCH_COMPLETED)
        def update_rg_metrices(engine: Engine) -> None:
            file_path = None
            self.rg_evaluator.print_stats(epoch_num=self.state.epoch, writer=self.writer,
                                          file_path=file_path)

        @self.on(Events.EPOCH_STARTED)
        def empty_buffers(engine: Engine) -> None:
            self.rg_evaluator.reset()


def build_evaluator(val_loader, net, optimizer, scheduler, writer, config, device, loss, distributed=False, local_rank=0,
                    **kwargs):
    """[summary]

    Args:
        val_loader ([type]): [description]
        net ([type]): [description]
        device ([type]): [description]

    Returns:
        [type]: [description]
    """
    val_handlers = [
        TensorBoardStatsHandler(
            writer,
            tag_name="val_smd",
            output_transform=lambda x: None,
            global_epoch_transform=lambda x: scheduler.last_epoch
        ),
    ]
    if local_rank == 0:
        val_handlers.extend(
            [
                StatsHandler(output_transform=lambda x: None),
                CheckpointSaver(
                    save_dir=os.path.join(config.TRAIN.SAVE_PATH, "runs",
                                          '%s_%d' % (config.log.exp_name, config.DATA.SEED),
                                          'models'),
                    save_dict={"net": net, "optimizer": optimizer, "scheduler": scheduler},
                    save_key_metric=False,
                    key_metric_n_saved=5,
                    save_interval=1
                ),
            ]
        )

    evaluator = RelationformerEvaluator(
        config=config,
        device=device,
        val_data_loader=val_loader,
        network=net,
        inferer=SimpleInferer(),
        val_handlers=val_handlers,
        amp=False,
        distributed=distributed,
        writer=writer,
        loss_function=loss,
        **kwargs,
    )

    return evaluator
