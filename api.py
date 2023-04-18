import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import json
import os
from timm.data.mixup import Mixup
from timm.models import create_model
from timm.utils import ModelEma
from optim_factory import create_optimizer, get_parameter_groups, \
    LayerDecayValueAssigner, get_is_head_flag_for_vit
from engine_for_finetuning import get_handler, evaluate
from datasets import create_downstream_hwj_test_dataset
from utils import NativeScalerWithGradNormCount as NativeScaler
import utils
import modeling_finetune

def set_args_from_dict(parser: argparse.ArgumentParser, args_dict: dict):
    for key, value in args_dict.items():
        action = parser.get_default(key)
        if action is not None:
            action.default = value
        else:
            parser.set_defaults(**{key: value})


class beit3_multi_labels_model:
    def get_args(self):
        # 加载参数字典
        with open('config.json', 'r') as f:
            loaded_args = json.load(f)
        parser = argparse.ArgumentParser()
        # 将参数设置回 `argparse.ArgumentParser` 对象中
        set_args_from_dict(parser, loaded_args)

        known_args, _ = parser.parse_known_args()

        if known_args.enable_deepspeed:
            try:
                import deepspeed
                from deepspeed import DeepSpeedConfig
                parser = deepspeed.add_config_arguments(parser)
                ds_init = deepspeed.initialize
            except:
                print("Please 'pip install deepspeed==0.4.0'")
                exit(0)
        else:
            ds_init = None

        return parser.parse_args(), ds_init

    def __init__(self):
        args, ds_init = self.get_args()
        utils.init_distributed_mode(args)
        if ds_init is not None:
            utils.create_ds_config(args)
        if args.task_cache_path is None:
            args.task_cache_path = args.output_dir
        print(args)

        device = torch.device(args.device)

        # fix the seed for reproducibility
        seed = args.seed + utils.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        # random.seed(seed)

        cudnn.benchmark = True

        if utils.get_rank() == 0 and args.log_dir is not None:
            os.makedirs(args.log_dir, exist_ok=True)
            log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
        else:
            log_writer = None

        if not args.model.endswith(args.task):
            if args.task in ("flickr30k", "coco_retrieval"):
                model_config = "%s_retrieval" % args.model
            elif args.task in ("coco_captioning", "nocaps"):
                model_config = "%s_captioning" % args.model
            elif args.task in ("imagenet"):
                model_config = "%s_imageclassification" % args.model
            else:
                model_config = "%s_%s" % (args.model, args.task)
        else:
            model_config = args.model

        print("model_config = %s" % model_config)
        model = create_model(
            model_config,
            pretrained=False,
            drop_path_rate=args.drop_path,
            vocab_size=args.vocab_size,
            checkpoint_activations=args.checkpoint_activations,
        )

        if args.checkpoint:
            model.load_state_dict(torch.load(args.checkpoint)['module'])

        model.to(device)

        model_ema = None
        if args.model_ema:
            # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
            model_ema = ModelEma(
                model,
                decay=args.model_ema_decay,
                device='cpu' if args.model_ema_force_cpu else '',
                resume='')
            print("Using EMA with decay = %.8f" % args.model_ema_decay)

        model_without_ddp = model
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print("Model = %s" % str(model_without_ddp))
        print('number of params:', n_parameters)

        total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()

        num_layers = model_without_ddp.get_num_layers()
        if args.layer_decay < 1.0:
            lrs = list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2))
            assigner = LayerDecayValueAssigner(lrs)
        elif args.task_head_lr_weight > 1:
            assigner = LayerDecayValueAssigner([1.0, args.task_head_lr_weight], scale_handler=get_is_head_flag_for_vit)
        else:
            assigner = None

        if assigner is not None:
            print("Assigned values = %s" % str(assigner.values))

        skip_weight_decay_list = model.no_weight_decay()

        if args.distributed:
            torch.distributed.barrier()
        if args.enable_deepspeed:
            loss_scaler = None
            optimizer_params = get_parameter_groups(
                model, args.weight_decay, skip_weight_decay_list,
                assigner.get_layer_id if assigner is not None else None,
                assigner.get_scale if assigner is not None else None)
            model, optimizer, _, _ = ds_init(
                args=args, model=model, model_parameters=optimizer_params,
                dist_init_required=not args.distributed,
            )

            print("model.gradient_accumulation_steps() = %d" % model.gradient_accumulation_steps())
            assert model.gradient_accumulation_steps() == args.update_freq
        else:
            if args.distributed:
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
                                                                  find_unused_parameters=True)
                model_without_ddp = model.module

            optimizer = create_optimizer(
                args, model_without_ddp, skip_list=skip_weight_decay_list,
                get_num_layer=assigner.get_layer_id if assigner is not None else None,
                get_layer_scale=assigner.get_scale if assigner is not None else None)
            loss_scaler = NativeScaler()

        utils.auto_load_model(
            args=args, model=model, model_without_ddp=model_without_ddp,
            optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

        task_handler = get_handler(args)

        data_loader_test = create_downstream_hwj_test_dataset(args, is_eval=True)

        ext_test_stats, task_key = evaluate(data_loader_test, model, device, task_handler)
        print(
            f"Accuracy of the network on the {len(data_loader_test.dataset)} test images: {ext_test_stats[task_key]:.3f}%")
        exit(0)

if __name__ == '__main__':
    beit3_multi_labels_model()