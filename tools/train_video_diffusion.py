from accelerate.utils import write_basic_config
write_basic_config()
import warnings

import logging
import os
import math
import shutil
from pathlib import Path
import numpy as np
from einops import rearrange
import accelerate
from collections import defaultdict

from tqdm.auto import tqdm
from peft import LoraConfig

import torch
torch.cuda.empty_cache()
import torch.nn.functional as F
import torch.utils.checkpoint

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from packaging import version
from diffusers.utils.import_utils import is_xformers_available
from torchvision.transforms.functional import to_pil_image

from diffusers import EulerDiscreteScheduler
from diffusers.models import AutoencoderKLTemporalDecoder
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils import is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.training_utils import EMAModel

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from ctrlv.utils import parse_args, get_dataloader, encode_video_image, get_add_time_ids, get_fourier_embeds_from_boundingbox, get_n_training_samples, wandb_frames_with_bbox, get_model_attr
    from ctrlv.models import UNetSpatioTemporalConditionModel#, UNetSpatioTemporalConditionModel_with_bbox_cond
    from ctrlv.pipelines import VideoDiffusionPipeline

if not is_wandb_available():
    warnings.warn("Make sure to install wandb if you want to use it for logging during training.")
else: 
    import wandb
logger = get_logger(__name__, log_level="INFO")

def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    try:
        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(accelerator.state, main_process_only=False)
        # If passed along, set the training seed now.
        if args.seed is not None:
            set_seed(args.seed)

        # Handle the repository creation
        if accelerator.is_main_process:
            if args.output_dir is not None:
                os.makedirs(args.output_dir, exist_ok=True)
                plot_dir = os.path.join(args.output_dir, "plots")
                os.makedirs(plot_dir, exist_ok=True)

        # Load scheduler, tokenizer and models.
        noise_scheduler = EulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        # Load scheduler, tokenizer and models.
        vae = AutoencoderKLTemporalDecoder.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant="fp16",
        )
        
        # Initialize DualVAEManager for semantic VAE support
        vae_manager = None
        if args.use_segmentation:
            from ctrlv.models import DualVAEManager
            semantic_vae_ckpt = "/usrhomes/s1492/vae_semantic/checkpoints/semantic_vae_native/best_model_with_dice_boundaryweight.pth"
            logger.info(f"Initializing DualVAEManager with semantic VAE from {semantic_vae_ckpt}")
            vae_manager = DualVAEManager(
                rgb_vae=vae,
                semantic_vae_checkpoint=semantic_vae_ckpt,
                num_semantic_classes=19,
                device=accelerator.device,
                clip_size=args.clip_length,  # Use same clip_length as training
                verbose=True
            )
            logger.info("✓ DualVAEManager initialized for semantic VAE encoding")
        # in_channels = 12 if args.add_bbox_frame_conditioning else 8
        # in_channels = 8
        # if args.add_bbox_frame_conditioning:
        #     unet = UNetSpatioTemporalConditionModel_with_bbox_cond.from_pretrained(
        #     args.pretrained_model_name_or_path, subfolder="unet", variant="fp16", revision=args.non_ema_revision,
        #     low_cpu_mem_usage=True, in_channels=in_channels, ignore_mismatched_sizes=True
        # )
        # else:
        unet = UNetSpatioTemporalConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", variant="fp16",
            low_cpu_mem_usage=True, num_frames=args.clip_length
        )

        feature_extractor = CLIPImageProcessor.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="feature_extractor", revision=args.revision,
        )

        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="image_encoder", revision=args.revision, variant="fp16"
        )

        # freeze parameters of models to save more memory
        vae.requires_grad_(False)
        image_encoder.requires_grad_(False)

        # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        
        # Move unet, vae and text_encoder to device and cast to weight_dtype
        vae.to(accelerator.device, dtype=weight_dtype)
        image_encoder.to(accelerator.device, dtype=weight_dtype)
        
        # Add LoRA adapter
        if args.enable_lora:
            unet.requires_grad_(False)
            unet_lora_config = LoraConfig(
                r=args.rank,
                lora_alpha=args.rank,
                init_lora_weights="gaussian",
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            )
            # Add adapter and make sure the trainable params are in float32.
            unet.add_adapter(unet_lora_config) # this line of code disable the unet add_embedding's gradient

        # # Create EMA for the unet.
        if args.use_ema:
            ema_unet = EMAModel(unet.parameters(), model_cls=UNetSpatioTemporalConditionModel, model_config=unet.config)
        
        if args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                import xformers

                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse("0.0.16"):
                    logger.warning(
                        "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                    )
                unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")
        
        # `accelerate` 0.16.0 will have better support for customized saving
        if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
            # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
            def save_model_hook(models, weights, output_dir):
                if accelerator.is_main_process:
                    if args.use_ema:
                        ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                    for i, model in enumerate(models):
                        model.save_pretrained(os.path.join(output_dir, "unet"), safe_serialization=False)

                        # make sure to pop weight so that corresponding model is not saved again
                        weights.pop()
            
            def load_model_hook(models, input_dir):
                if args.use_ema:
                    load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNetSpatioTemporalConditionModel)
                    ema_unet.load_state_dict(load_model.state_dict())
                    ema_unet.to(accelerator.device)
                    del load_model

                for _ in range(len(models)):
                    # pop models so that they are not loaded again
                    model = models.pop()

                    # load diffusers style into model
                    # if args.add_bbox_frame_conditioning:
                    #     load_model = UNetSpatioTemporalConditionModel_with_bbox_cond.from_pretrained(input_dir, subfolder="unet")
                    # else:
                    load_model = UNetSpatioTemporalConditionModel.from_pretrained(input_dir, subfolder="unet")
                    model.register_to_config(**load_model.config)

                    model.load_state_dict(load_model.state_dict())
                    del load_model

            accelerator.register_save_state_pre_hook(save_model_hook)
            accelerator.register_load_state_pre_hook(load_model_hook)


        if args.enable_gradient_checkpointing:
            unet.enable_gradient_checkpointing()

        if args.scale_lr:
            args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
            )
        
        if_only_temporal_backprop = False
        if args.add_bbox_frame_conditioning:
            unet.enable_grad_bbox_frame_embedder()
        if args.enable_lora:
            pass
        elif args.backprop_temporal_blocks_start_iter == 0:
            parameters_list = unet.enable_grad(temporal_transformer_block=True, all=False)
            if_only_temporal_backprop = True
        else:
            parameters_list = unet.enable_grad(all=True)
        parameters_list = unet.get_parameters_with_grad()

        if args.mixed_precision == "fp16":
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(unet, dtype=torch.float32)
                
        optimizer = torch.optim.AdamW(
            parameters_list, 
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

        train_dataset, train_loader = get_dataloader(args.data_root, args.dataset_name, if_train=True, clip_length=args.clip_length,
                                                    batch_size=args.train_batch_size, num_workers=args.dataloader_num_workers, 
                                                    data_type='clip', use_default_collate=True, tokenizer=None, shuffle=True,
                                                    if_return_bbox_im=True, train_H=args.train_H, train_W=args.train_W,
                                                    use_segmentation=args.use_segmentation, 
                                                    use_preplotted_bbox=not args.if_last_frame_trajectory,
                                                    if_last_frame_traj=args.if_last_frame_trajectory,
                                                    non_overlapping_clips=args.non_overlapping_clips,
                                                    return_semantic_ids=args.use_segmentation)
        # _, test_loader = get_dataloader(args.dataset_name, if_train=True, 
        #                                 batch_size=1, num_workers=args.dataloader_num_workers, 
        #                                 data_type='clip', use_default_collate=True, tokenizer=None, shuffle=True)
        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True

        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
        )

        # Prepare everything with our `accelerator`.
        unet, optimizer, train_loader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_loader, lr_scheduler
        )
        if args.use_ema:
            ema_unet.to(accelerator.device)

        # unet, optimizer, train_loader, test_loader, lr_scheduler = accelerator.prepare(
        #     unet, optimizer, train_loader, test_loader, lr_scheduler
        # )

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
        if overrode_max_train_steps:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if accelerator.is_main_process:
            accelerator.init_trackers(args.project_name, config=vars(args), init_kwargs={"wandb": {"dir": args.output_dir, "name": args.run_name, "entity": args.wandb_entity}})

        def get_sigmas(timesteps, n_dim=5, dtype=torch.float32):
            sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
            schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
            timesteps = timesteps.to(accelerator.device)

            step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

            sigma = sigmas[step_indices].flatten()
            while len(sigma.shape) < n_dim:
                sigma = sigma.unsqueeze(-1)
            return sigma
        
        # Train!
        total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        initial_global_step = global_step = 0
        first_epoch = 0

        # # Potentially load in the weights and states from a previous save
        if args.resume_from_checkpoint:
            if args.resume_from_checkpoint != "latest":
                path = os.path.basename(args.resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = os.listdir(args.output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None

            if path is None:
                accelerator.print(
                    f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
                )
                args.resume_from_checkpoint = None
                initial_global_step = 0
            else:
                accelerator.print(f"Resuming from checkpoint {path}")
                accelerator.load_state(os.path.join(args.output_dir, path))
                initial_global_step = global_step = int(path.split("-")[1])

                first_epoch = global_step // num_update_steps_per_epoch
        else:
            initial_global_step = 0

        progress_bar = tqdm(
            range(0, args.max_train_steps),
            initial=initial_global_step,
            desc="Steps",
            # Only show the progress bar once on each machine.
            disable=not accelerator.is_local_main_process,
        )

        def unwrap_model(model):
            model = accelerator.unwrap_model(model)
            model = model._orig_mod if is_compiled_module(model) else model
            return model

        demo_samples = get_n_training_samples(train_loader, args.num_demo_samples)

        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None

        def run_inference_with_pipeline(pipeline, demo_samples, log_dict):

            for sample_i, sample in enumerate(demo_samples):
                if args.if_last_frame_trajectory:
                    sample_bbox = sample['bbox_img'][:args.clip_length]
                    sample_bbox[-1] = sample['bbox_img'][args.clip_length]
                
                # Prepare semantic conditioning if using semantic VAE
                semantic_ids_cond = None
                use_sem_vae = False
                if args.predict_bbox and args.use_segmentation and vae_manager is not None and 'semantic_ids' in sample:
                    semantic_ids_cond = sample['semantic_ids'].unsqueeze(0)  # [1, T, H, W]
                    use_sem_vae = True
                
                # Run diffusion inference
                result = pipeline(sample['image_init'], 
                                height=train_dataset.train_H, width=train_dataset.train_W, 
                                # bbox_conditions=sample['objects_tensors'], original_size=(train_dataset.orig_W, train_dataset.orig_H),
                                bbox_images=sample['bbox_img'].unsqueeze(0) if args.predict_bbox else None,
                                decode_chunk_size=8, motion_bucket_id=127, fps=args.fps, 
                                num_inference_steps=args.num_inference_steps,
                                num_frames=args.clip_length,
                                min_guidance_scale=args.min_guidance_scale,
                                max_guidance_scale=args.max_guidance_scale,
                                noise_aug_strength=args.noise_aug_strength,
                                generator=generator, 
                                output_type='latent' if (args.predict_bbox and args.use_segmentation and vae_manager is not None) else 'pt',
                                num_cond_bbox_frames=args.num_cond_bbox_frames,
                                semantic_ids=semantic_ids_cond,
                                use_semantic_vae=use_sem_vae)
                
                # Handle output based on whether we're using semantic VAE
                if args.predict_bbox and args.use_segmentation and vae_manager is not None:
                    # Stage 1: Decode semantic latents to semantic IDs
                    latents = result.frames[0]  # [T, C, H, W]
                    latents_flat = rearrange(latents, "f c h w -> f c h w")
                    
                    # Decode semantic latents using Semantic VAE
                    semantic_ids = vae_manager.decode_semantic(latents_flat)  # [T, H, W] trainIDs 0-18
                    
                    # Convert trainIDs back to original KITTI-360 IDs
                    from ctrlv.utils.semantic_preprocessing import KITTI360_LABEL_MAPPING, semantic_ids_to_viz_rgb
                    trainid_to_original = {train_id: kitti_id for kitti_id, train_id in KITTI360_LABEL_MAPPING.items()}
                    semantic_ids_original = semantic_ids.clone()
                    for train_id, orig_id in trainid_to_original.items():
                        semantic_ids_original[semantic_ids == train_id] = orig_id
                    
                    # Convert to RGB colormap for WandB visualization [T, H, W] -> [T, 3, H, W]
                    semantic_ids_np = semantic_ids.cpu().numpy()  # [T, H, W]
                    T, H, W = semantic_ids_np.shape
                    frames_rgb = np.zeros((T, 3, H, W), dtype=np.uint8)
                    for t in range(T):
                        rgb_frame = semantic_ids_to_viz_rgb(semantic_ids_np[t])  # [H, W] -> [H, W, 3]
                        frames_rgb[t] = rgb_frame.transpose(2, 0, 1)  # [3, H, W]
                    
                    # Log colorful RGB semantic frames for Stage 1
                    log_dict["generated_semantic_frames"].append(wandb.Video(frames_rgb, fps=args.fps))

                    # Convert GT trainIDs to RGB colormap for visualization
                    # Use semantic_ids [T, H, W] not bbox_img [T, 3, H, W] which is RGB visualization
                    if 'semantic_ids' in sample:
                        gt_semantic_ids = sample['semantic_ids']  # [T, H, W] trainIDs 0-18
                        gt_semantic_np = gt_semantic_ids.cpu().numpy()

                        T_gt, H_gt, W_gt = gt_semantic_np.shape
                        gt_frames_rgb = np.zeros((T_gt, 3, H_gt, W_gt), dtype=np.uint8)
                        for t in range(T_gt):
                            gt_rgb_frame = semantic_ids_to_viz_rgb(gt_semantic_np[t])  # [H, W] -> [H, W, 3]
                            gt_frames_rgb[t] = gt_rgb_frame.transpose(2, 0, 1)  # [3, H, W]

                        log_dict["gt_semantic_frames"].append(wandb.Video(gt_frames_rgb, fps=args.fps))

                        # Debug mIoU: compare predicted vs GT semantic IDs
                        pred_np = semantic_ids_np  # [T, H, W] already computed above
                        gt_np = gt_semantic_np     # [T, H, W]
                        # Resize pred to GT if shapes differ
                        if pred_np.shape != gt_np.shape:
                            import torch.nn.functional as F_resize
                            pred_t = torch.from_numpy(pred_np).unsqueeze(1).float()
                            pred_t = F_resize.interpolate(pred_t, size=(gt_np.shape[1], gt_np.shape[2]), mode='nearest')
                            pred_np = pred_t.squeeze(1).numpy().astype(np.int64)
                        # Per-class IoU via confusion matrix
                        num_cls = 19
                        valid_mask = (gt_np >= 0) & (gt_np < num_cls) & (pred_np >= 0) & (pred_np < num_cls)
                        if valid_mask.any():
                            gt_valid = gt_np[valid_mask]
                            pred_valid = pred_np[valid_mask]
                            conf = np.zeros((num_cls, num_cls), dtype=np.int64)
                            np.add.at(conf, (gt_valid, pred_valid), 1)
                            intersection = np.diag(conf)
                            union = conf.sum(axis=1) + conf.sum(axis=0) - intersection
                            valid_classes = union > 0
                            iou_per_class = np.where(valid_classes, intersection / (union + 1e-10), 0.0)
                            sample_miou = iou_per_class[valid_classes].mean() if valid_classes.any() else 0.0
                            pixel_acc = intersection.sum() / (conf.sum() + 1e-10)
                            log_dict.setdefault("val_miou_per_sample", []).append(sample_miou)
                            log_dict.setdefault("val_pixel_acc_per_sample", []).append(pixel_acc)
                    else:
                        logger.warning("semantic_ids not found in sample, skipping gt_semantic_frames logging")
                else:
                    # Stage 2 or RGB mode: Normal RGB decoding
                    frames = result.frames[0]
                    frames = frames.detach().cpu().numpy()*255
                    frames = frames.astype(np.uint8)
                    log_dict["generated_videos"].append(wandb.Video(frames, fps=args.fps))
                    log_dict["gt_videos"].append(wandb.Video(sample['gt_clip_np'], fps=args.fps))
                    frame_bboxes = wandb_frames_with_bbox(frames, sample['objects_tensors'], (train_dataset.orig_W, train_dataset.orig_H))
                    log_dict["frames_with_bboxes_{}".format(sample_i)] = frame_bboxes
            return log_dict

        # ── Early stopping ────────────────────────────────────────────────────
        # State is written to output_dir (not inside checkpoint-XXXXX), so it
        # survives --resume_from_checkpoint latest across SLURM re-submissions.
        # Patience is in units of validation events (each = validation_steps steps).
        EARLY_STOP_PATIENCE   = 6      # stop after 6 validations without improvement
        EARLY_STOP_MIN_DELTA  = 0.002  # require at least 0.2% absolute mIoU gain
        _es_state_path = os.path.join(args.output_dir, 'early_stop_state.json')

        def _early_stop_check(metric_value, step):
            """Read/update early_stop_state.json; return (patience_counter, should_stop)."""
            import json as _json
            if os.path.exists(_es_state_path):
                with open(_es_state_path) as _f:
                    _s = _json.load(_f)
            else:
                _s = {"best_metric": -1.0, "best_step": 0,
                      "patience_counter": 0, "history": []}
            _s["history"].append({"step": step, "metric": metric_value})
            if metric_value > _s["best_metric"] + EARLY_STOP_MIN_DELTA:
                _s["best_metric"] = metric_value
                _s["best_step"]   = step
                _s["patience_counter"] = 0
                # Copy current checkpoint as best_checkpoint
                _ckpts = sorted(
                    [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")],
                    key=lambda x: int(x.split("-")[1])
                )
                if _ckpts:
                    _src = os.path.join(args.output_dir, _ckpts[-1])
                    _dst = os.path.join(args.output_dir, "best_checkpoint")
                    if os.path.exists(_dst):
                        shutil.rmtree(_dst)
                    shutil.copytree(_src, _dst)
                    logger.info(f"Saved best_checkpoint (mIoU={metric_value:.4f} @ step {step})")
            else:
                _s["patience_counter"] += 1
            with open(_es_state_path, 'w') as _f:
                _json.dump(_s, _f, indent=2)
            return _s["patience_counter"], _s["patience_counter"] >= EARLY_STOP_PATIENCE, \
                   _s["best_metric"], _s["best_step"]

        early_stop_triggered = False

        # Latent statistics tracker for scaling factor analysis
        # Tracks running mean/std of semantic latents to compare with RGB VAE scaling factor (0.18215)
        latent_stats = {
            'sum': 0.0, 'sq_sum': 0.0, 'count': 0,
            'min': float('inf'), 'max': float('-inf'),
            'per_channel_sum': np.zeros(4), 'per_channel_sq_sum': np.zeros(4), 'per_channel_count': 0,
        }

        for epoch in range(first_epoch, args.num_train_epochs):

            train_loss = 0.0
            for _, batch in enumerate(train_loader):

                unet_dtype = get_model_attr(unet, 'dtype') 
                
                ## update optimizer's parameters
                if args.backprop_temporal_blocks_start_iter >= 0:
                    if not if_only_temporal_backprop and global_step >= args.backprop_temporal_blocks_start_iter:
                        optimizer.param_groups.clear()
                        optimizer.state.clear()
                        parameters_list = unet.enable_grad(temporal_transformer_block=True, all=False)
                        parameters_list = unet.get_parameters_with_grad()
                        optimizer.add_param_group({"params": parameters_list})
                        if_only_temporal_backprop = True
                        logger.info("Start only backpropagating the temporal layers.")

                if accelerator.sync_gradients:
                    if accelerator.is_main_process:
                        
                        if global_step % args.validation_steps == 0:
                            logger.info("Running validation... ")

                            log_dict = defaultdict(list)
                            with torch.autocast(
                                    str(accelerator.device).replace(":0", ""), enabled=accelerator.mixed_precision == "fp16"
                                ):
                                # create pipeline
                                if args.use_ema:
                                    # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                                    ema_unet.store(unet.parameters())
                                    ema_unet.copy_to(unet.parameters())
                                pipeline = VideoDiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path,
                                                                            unet=unwrap_model(unet), 
                                                                            revision=args.revision, 
                                                                            variant=args.variant, 
                                                                            torch_dtype=weight_dtype,
                                                                            feature_extractor=feature_extractor,
                                                                            image_encoder=unwrap_model(image_encoder),
                                                                            vae=unwrap_model(vae),)
                                # Attach vae_manager for semantic VAE decoding in Stage 1
                                if vae_manager is not None:
                                    pipeline.vae_manager = vae_manager
                                unet.eval()
                                pipeline = pipeline.to(accelerator.device)
                                pipeline.set_progress_bar_config(disable=True)
                                log_dict = run_inference_with_pipeline(pipeline, demo_samples, log_dict)

                            # Aggregate and log mIoU metrics if computed
                            if "val_miou_per_sample" in log_dict:
                                miou_samples = log_dict.pop("val_miou_per_sample")
                                pixacc_samples = log_dict.pop("val_pixel_acc_per_sample")
                                log_dict["val/miou"] = float(np.mean(miou_samples))
                                log_dict["val/pixel_accuracy"] = float(np.mean(pixacc_samples))
                                logger.info(f"Validation mIoU: {log_dict['val/miou']:.4f}, Pixel Acc: {log_dict['val/pixel_accuracy']:.4f}")

                                # Early stopping (resume-safe via persistent state file)
                                _pc, _stop, _best, _best_step = _early_stop_check(
                                    log_dict["val/miou"], global_step
                                )
                                log_dict["early_stop/patience_counter"] = _pc
                                log_dict["early_stop/best_miou"]        = _best
                                logger.info(
                                    f"Early stop: patience {_pc}/{EARLY_STOP_PATIENCE}, "
                                    f"best mIoU {_best:.4f} @ step {_best_step}"
                                )
                                if _stop:
                                    logger.info(
                                        f"Early stopping triggered! No improvement for "
                                        f"{EARLY_STOP_PATIENCE} validations. "
                                        f"Best mIoU={_best:.4f} @ step {_best_step}"
                                    )
                                    early_stop_triggered = True

                            for tracker in accelerator.trackers:
                                if tracker.name == "wandb":
                                    tracker.log(log_dict)
                            if args.use_ema:
                                # Switch back to the original UNet parameters.
                                ema_unet.restore(unet.parameters())

                            del pipeline, log_dict
                            torch.cuda.empty_cache()
                
                unet.train()
                with accelerator.accumulate(unet):
                    # Forward pass
                    batch_size, video_length = batch['clips'].shape[0], batch['clips'].shape[1]
                    initial_images = batch['clips'][:,0,:,:,:] # only use the first frame

                    # Encode input image
                    encoder_hidden_states = encode_video_image(initial_images, feature_extractor, weight_dtype, image_encoder).unsqueeze(1)
                    
                    # Encode bbox objects
                    # encoded_objects = get_fourier_embeds_from_boundingbox(batch['objects'], (train_dataset.orig_W, train_dataset.orig_H), dropout_prob=args.bbox_dropout_prob, generator=generator,)

                    # Encode clip frames using VAE (RGB or Semantic)
                    # [batch, frames, channels, height, width] -> [batch*frames, channels, height, width]
                    if args.predict_bbox and args.use_segmentation and vae_manager is not None:
                        assert batch.get('semantic_ids') is not None, \
                            "semantic_ids not found in batch! Ensure return_semantic_ids=True is passed to dataset."
                    if args.predict_bbox and args.use_segmentation and vae_manager is not None and batch.get('semantic_ids') is not None:
                        # Use Semantic VAE for semantic ID encoding (grayscale trainIDs)
                        semantic_ids = rearrange(batch['semantic_ids'], "b f h w -> (b f) h w")
                        latents = vae_manager.encode_semantic_from_ids(semantic_ids)
                        latents = rearrange(latents, "(b f) c h w -> b f c h w", b=batch_size)
                        if global_step == 0 and accelerator.is_main_process:
                            logger.info("Using Semantic VAE encoding for target latents (semantic mode active)")
                    else:
                        # Use RGB VAE for RGB image encoding
                        frames = rearrange(batch['clips'] if not args.predict_bbox else batch['bbox_images'], "b f c h w -> (b f) c h w")
                        latents = vae.encode(frames.to(dtype=weight_dtype)).latent_dist.sample()
                        latents = rearrange(latents, "(b f) c h w -> b f c h w", b=batch_size)
                    
                    # Encode initial frame for conditioning (frames 1-23)
                    # IMPORTANT: Must use the SAME VAE as target latents to keep
                    # conditioning in the same latent space.
                    if args.predict_bbox and args.use_segmentation and vae_manager is not None and batch.get('semantic_ids') is not None:
                        # Use Semantic VAE: encode first frame's semantic IDs
                        first_frame_sem_ids = batch['semantic_ids'][:, 0, :, :]  # [B, H, W]
                        initial_frame_latent = vae_manager.encode_semantic_from_ids(first_frame_sem_ids)
                    else:
                        initial_frame_latent = vae.encode(initial_images.to(weight_dtype)).latent_dist.sample()
                    if not args.predict_bbox:
                        # Encode input image using VAE
                        conditional_latents = initial_frame_latent.to(dtype=unet_dtype)
                    else:
                        if args.if_last_frame_trajectory:
                            conditional_latents = latents.clone().to(dtype=unet_dtype)
                            last_conditional_latents = conditional_latents[:,-1,::]
                            latents = latents[:,:-1,::]
                            conditional_latents = conditional_latents[:,:-1,::]
                            conditional_latents[:,args.num_cond_bbox_frames:-1,::] = initial_frame_latent.unsqueeze(1).repeat(1, video_length-args.num_cond_bbox_frames-1, 1, 1, 1)
                            conditional_latents[:,-1,::] = last_conditional_latents
                        else:
                            conditional_latents = latents.clone().to(dtype=unet_dtype)
                            conditional_latents[:,args.num_cond_bbox_frames:-1,::] = initial_frame_latent.unsqueeze(1).repeat(1, video_length-args.num_cond_bbox_frames-1, 1, 1, 1)

                    # ---- Latent statistics tracking (lightweight, no GPU sync) ----
                    # Collect stats on UNSCALED latents to evaluate whether 0.18215 is appropriate
                    if accelerator.is_main_process:
                        with torch.no_grad():
                            lat_detached = latents.detach().float()
                            lat_mean = lat_detached.mean().item()
                            lat_std = lat_detached.std().item()
                            lat_min = lat_detached.min().item()
                            lat_max = lat_detached.max().item()
                            numel = lat_detached.numel()

                            latent_stats['sum'] += lat_mean * numel
                            latent_stats['sq_sum'] += (lat_std**2 + lat_mean**2) * numel
                            latent_stats['count'] += numel
                            latent_stats['min'] = min(latent_stats['min'], lat_min)
                            latent_stats['max'] = max(latent_stats['max'], lat_max)

                            # Per-channel stats [B, F, C, H, W] -> mean over B, F, H, W
                            per_ch_mean = lat_detached.mean(dim=(0, 1, 3, 4)).cpu().numpy()  # [4]
                            per_ch_std = lat_detached.std(dim=(0, 1, 3, 4)).cpu().numpy()    # [4]
                            n_spatial = lat_detached.shape[0] * lat_detached.shape[1] * lat_detached.shape[3] * lat_detached.shape[4]
                            latent_stats['per_channel_sum'] += per_ch_mean * n_spatial
                            latent_stats['per_channel_sq_sum'] += (per_ch_std**2 + per_ch_mean**2) * n_spatial
                            latent_stats['per_channel_count'] += n_spatial

                        # One-time detailed report at step 0
                        if global_step == 0:
                            suggested_sf = 1.0 / lat_std if lat_std > 0 else 0.18215
                            logger.info(
                                f"\n{'='*60}\n"
                                f"LATENT STATISTICS REPORT (Step 0, first batch)\n"
                                f"{'='*60}\n"
                                f"  Unscaled semantic latents:\n"
                                f"    Mean:  {lat_mean:.6f}\n"
                                f"    Std:   {lat_std:.6f}\n"
                                f"    Min:   {lat_min:.6f}\n"
                                f"    Max:   {lat_max:.6f}\n"
                                f"  Per-channel mean: {per_ch_mean}\n"
                                f"  Per-channel std:  {per_ch_std}\n"
                                f"  Current scaling factor (RGB VAE): {vae.config.scaling_factor}\n"
                                f"  Suggested scaling factor (1/std): {suggested_sf:.6f}\n"
                                f"  Ratio (suggested/current):        {suggested_sf / vae.config.scaling_factor:.2f}x\n"
                                f"  After scaling with 0.18215:\n"
                                f"    Scaled std: {lat_std * vae.config.scaling_factor:.6f}\n"
                                f"    (ideal ≈ 1.0 for noise schedule)\n"
                                f"{'='*60}"
                            )

                    target_latents = latents = latents * vae.config.scaling_factor

                    # Clean up memory - only delete frames if it was created (RGB VAE path)
                    del batch
                    if 'frames' in locals():
                        del frames
                    noise = torch.randn_like(latents)
                    
                    indices = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,)).to(noise_scheduler.timesteps).long()
                    timesteps = noise_scheduler.timesteps[indices].to(accelerator.device)

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    # Scale the noisy latents for the UNet
                    sigmas = get_sigmas(timesteps, len(noisy_latents.shape), noisy_latents.dtype)
                    # inp_noisy_latents = noise_scheduler.scale_model_input(noisy_latents, timesteps)
                    inp_noisy_latents = noisy_latents / ((sigmas**2 + 1) ** 0.5)

                    added_time_ids = get_add_time_ids(
                        fps=args.fps-1,
                        motion_bucket_id=127,
                        noise_aug_strength=args.noise_aug_strength,
                        dtype=weight_dtype,
                        batch_size=batch_size,
                        unet=unet
                    ).to(accelerator.device)
                    
                    # Conditioning dropout to support classifier-free guidance during inference. For more details
                    # check out the section 3.2.1 of the original paper https://arxiv.org/abs/2211.09800.
                    # Addapted from https://github.com/huggingface/diffusers/blob/0d2d424fbef933e4b81bea20a660ee6fc8b75ab0/docs/source/en/training/instructpix2pix.md
                    if args.conditioning_dropout_prob is not None:
                        random_p = torch.rand(
                            batch_size, device=accelerator.device, generator=generator)
                        # Sample masks for the edit prompts.
                        prompt_mask = random_p < 2 * args.conditioning_dropout_prob
                        prompt_mask = prompt_mask.reshape(batch_size, 1, 1)
                        # Final text conditioning.
                        null_conditioning = torch.zeros_like(encoder_hidden_states)
                        encoder_hidden_states = torch.where(
                            prompt_mask, null_conditioning, encoder_hidden_states)
                        # Sample masks for the original images.
                        image_mask_dtype = conditional_latents.dtype
                        image_mask = 1 - (
                            (random_p >= args.conditioning_dropout_prob).to(
                                image_mask_dtype)
                            * (random_p < 3 * args.conditioning_dropout_prob).to(image_mask_dtype)
                        )
                        image_mask = image_mask.reshape(batch_size, 1, 1, 1, 1)
                        # Final image conditioning.
                        conditional_latents = image_mask * conditional_latents

                    # Concatenate the `original_image_embeds` with the `noisy_latents`.
                    if not args.predict_bbox:
                        conditional_latents = get_model_attr(unet, 'encode_bbox_frame')(conditional_latents, None)
                    
                    concatenated_noisy_latents = torch.cat([inp_noisy_latents, conditional_latents], dim=2)
                    model_pred = unet(concatenated_noisy_latents,
                                    timestep=timesteps,
                                    encoder_hidden_states=encoder_hidden_states.to(dtype=unet_dtype), 
                                    added_time_ids=added_time_ids.to(dtype=unet_dtype)).sample

                    # Denoise the latents
                    c_out = -sigmas / ((sigmas**2 + 1)**0.5)
                    c_skip = 1 / (sigmas**2 + 1)
                    denoised_latents = model_pred * c_out + c_skip * noisy_latents
                    weighting = (1 + sigmas ** 2) * (sigmas**-2.0)

                    # # MSE loss
                    loss = torch.mean(
                        (weighting.float() * (denoised_latents.float() - target_latents.float()) ** 2).reshape(target_latents.shape[0], -1),
                        dim=1,
                    )
                    loss = loss.mean()

                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                    train_loss += avg_loss.item() / args.gradient_accumulation_steps

                    # Backpropagate
                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                
                logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                del loss, latents, concatenated_noisy_latents, model_pred, weighting, inp_noisy_latents, noisy_latents, timesteps, indices, sigmas, added_time_ids

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    if args.use_ema:
                        ema_unet.step(get_model_attr(unet, 'parameters')())
                    progress_bar.update(1)
                    global_step += 1
                    log_plot = {
                                "train_loss": train_loss,
                                "lr": lr_scheduler.get_last_lr()[0],
                            }
                    if args.add_bbox_frame_conditioning:
                        log_plot["|attn_rz_weight|"] = get_model_attr(unet, 'get_attention_rz_weight')()

                    # Log latent statistics every 100 steps (negligible overhead)
                    if accelerator.is_main_process and latent_stats['count'] > 0 and global_step % 100 == 0:
                        running_mean = latent_stats['sum'] / latent_stats['count']
                        running_var = latent_stats['sq_sum'] / latent_stats['count'] - running_mean**2
                        running_std = max(running_var, 0.0) ** 0.5
                        suggested_sf = 1.0 / running_std if running_std > 0 else 0.18215
                        scaled_std = running_std * vae.config.scaling_factor

                        log_plot["latent_stats/unscaled_mean"] = running_mean
                        log_plot["latent_stats/unscaled_std"] = running_std
                        log_plot["latent_stats/unscaled_min"] = latent_stats['min']
                        log_plot["latent_stats/unscaled_max"] = latent_stats['max']
                        log_plot["latent_stats/scaled_std"] = scaled_std
                        log_plot["latent_stats/suggested_scaling_factor"] = suggested_sf
                        log_plot["latent_stats/current_scaling_factor"] = vae.config.scaling_factor

                        # Per-channel stats
                        if latent_stats['per_channel_count'] > 0:
                            ch_mean = latent_stats['per_channel_sum'] / latent_stats['per_channel_count']
                            ch_var = latent_stats['per_channel_sq_sum'] / latent_stats['per_channel_count'] - ch_mean**2
                            ch_std = np.sqrt(np.maximum(ch_var, 0.0))
                            for c in range(4):
                                log_plot[f"latent_stats/ch{c}_mean"] = float(ch_mean[c])
                                log_plot[f"latent_stats/ch{c}_std"] = float(ch_std[c])

                    accelerator.log(log_plot, step=global_step)
                    train_loss = 0.0

                    if global_step % args.checkpointing_steps == 0:
                        if accelerator.is_main_process:
                            # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                            if args.checkpoints_total_limit is not None:
                                checkpoints = os.listdir(args.output_dir)
                                checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                                # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                                if len(checkpoints) >= args.checkpoints_total_limit:
                                    num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                    removing_checkpoints = checkpoints[0:num_to_remove]

                                    logger.info(
                                        f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                    )
                                    logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                    for removing_checkpoint in removing_checkpoints:
                                        removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                        shutil.rmtree(removing_checkpoint)

                            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                            accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")

                if global_step >= args.max_train_steps or early_stop_triggered:
                    break

            if early_stop_triggered:
                logger.info("Exiting epoch loop due to early stopping.")
                break

        # Final latent statistics report
        accelerator.wait_for_everyone()
        if accelerator.is_main_process and latent_stats['count'] > 0:
            running_mean = latent_stats['sum'] / latent_stats['count']
            running_var = latent_stats['sq_sum'] / latent_stats['count'] - running_mean**2
            running_std = max(running_var, 0.0) ** 0.5
            suggested_sf = 1.0 / running_std if running_std > 0 else 0.18215
            scaled_std = running_std * vae.config.scaling_factor

            ch_mean = latent_stats['per_channel_sum'] / max(latent_stats['per_channel_count'], 1)
            ch_var = latent_stats['per_channel_sq_sum'] / max(latent_stats['per_channel_count'], 1) - ch_mean**2
            ch_std = np.sqrt(np.maximum(ch_var, 0.0))

            report = (
                f"\n{'='*70}\n"
                f"FINAL LATENT STATISTICS REPORT (accumulated over {global_step} steps)\n"
                f"{'='*70}\n"
                f"  Unscaled semantic latent statistics:\n"
                f"    Global mean:  {running_mean:.6f}\n"
                f"    Global std:   {running_std:.6f}\n"
                f"    Global min:   {latent_stats['min']:.6f}\n"
                f"    Global max:   {latent_stats['max']:.6f}\n"
                f"  Per-channel mean: [{', '.join(f'{m:.4f}' for m in ch_mean)}]\n"
                f"  Per-channel std:  [{', '.join(f'{s:.4f}' for s in ch_std)}]\n"
                f"\n"
                f"  Scaling factor analysis:\n"
                f"    Current (RGB VAE):   {vae.config.scaling_factor}\n"
                f"    Suggested (1/std):   {suggested_sf:.6f}\n"
                f"    Ratio (sugg/curr):   {suggested_sf / vae.config.scaling_factor:.2f}x\n"
                f"    Scaled std (curr):   {scaled_std:.6f}  (ideal ≈ 1.0)\n"
                f"    Scaled std (sugg):   {running_std * suggested_sf:.6f}\n"
                f"\n"
                f"  Interpretation:\n"
            )
            if abs(scaled_std - 1.0) < 0.3:
                report += f"    ✓ Current scaling factor is REASONABLE (scaled_std={scaled_std:.3f}, close to 1.0)\n"
            elif scaled_std < 0.7:
                report += (
                    f"    ⚠ Latents are OVER-COMPRESSED by current scaling factor\n"
                    f"      scaled_std={scaled_std:.3f} << 1.0 → noise dominates signal\n"
                    f"      Consider using suggested_sf={suggested_sf:.6f} for better training\n"
                )
            else:
                report += (
                    f"    ⚠ Latents are UNDER-COMPRESSED by current scaling factor\n"
                    f"      scaled_std={scaled_std:.3f} >> 1.0 → signal dominates noise\n"
                    f"      Consider using suggested_sf={suggested_sf:.6f} for better training\n"
                )
            report += f"{'='*70}"
            logger.info(report)

            # Save report to file
            report_path = os.path.join(args.output_dir, "latent_statistics_report.txt")
            with open(report_path, 'w') as f:
                f.write(report)
            logger.info(f"Latent statistics report saved to {report_path}")

        if accelerator.is_main_process:
            unet = unwrap_model(unet)
            if args.use_ema:
                ema_unet.copy_to(unet.parameters())
            pipeline = VideoDiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path,
                                                                unet=unet, 
                                                                revision=args.revision, 
                                                                variant=args.variant, 
                                                                torch_dtype=weight_dtype,
                                                                feature_extractor=feature_extractor,
                                                                image_encoder=unwrap_model(image_encoder),
                                                                vae=unwrap_model(vae),)
            pipeline.save_pretrained(args.output_dir)

            # Run a final round of inference
            logger.info("Running inference before terminating...")
            pipeline = pipeline.to(accelerator.device)
            pipeline.torch_dtype = weight_dtype
            pipeline.set_progress_bar_config(disable=True)

            log_dict = defaultdict(list)
            log_dict = run_inference_with_pipeline(pipeline, demo_samples, log_dict)
            for tracker in accelerator.trackers:
                if tracker.name == "wandb":
                    tracker.log(log_dict)
        
        accelerator.end_training()

    except KeyboardInterrupt:
        accelerator.end_training()
        if is_wandb_available():
            wandb.finish()
        torch.cuda.empty_cache()
        print("Keboard interrupt: shutdown requested... Exiting.")
        exit()
    except Exception:
        import sys, traceback
        traceback.print_exc(file=sys.stdout)
        if is_wandb_available():
            wandb.finish()
        torch.cuda.empty_cache()
        sys.exit(0)

if __name__ == '__main__':
    
    main()
    