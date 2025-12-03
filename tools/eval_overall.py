from accelerate.utils import write_basic_config
write_basic_config()
import warnings
from diffusers.utils import is_wandb_available

import logging, os
from pathlib import Path
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision.transforms.functional import to_pil_image
import wandb
from collections import defaultdict
from numpy import inf

import torch
torch.cuda.empty_cache()
import torch.utils.checkpoint

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from ctrlv.utils import parse_args, get_dataloader, eval_samples_generator, eval_demo_samples_generator, wandb_frames_with_bbox
    from ctrlv.pipelines import VideoDiffusionPipeline, StableVideoControlPipeline

logger = get_logger(__name__, log_level="INFO")

def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    accelerator.init_trackers(args.project_name, config=vars(args), init_kwargs={"wandb": {"dir": args.output_dir, "name": args.run_name}})
    wandb.define_metric("inference_step")

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

        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None

        dataset, data_loader = get_dataloader(args.data_root, args.dataset_name, if_train=False, clip_length=args.clip_length,
                                                    batch_size=args.train_batch_size, num_workers=args.dataloader_num_workers, 
                                                    data_type='clip', use_default_collate=True, tokenizer=None, shuffle=False,
                                                    if_return_bbox_im=True, train_H=args.train_H, train_W=args.train_W,
                                                    use_segmentation=args.use_segmentation, 
                                                    use_preplotted_bbox=not args.if_last_frame_trajectory,
                                                    if_last_frame_traj=args.if_last_frame_trajectory,)
        sample_generator = eval_samples_generator(data_loader)

        def run_pipelines(bbox_pipeline, ctrl_pipeline, sample_generator, one_step_limit=1):
            
            from ctrlv.metrics import f_measure, binary_mask_iou
            # from ignite.engine import create_supervised_evaluator

            metric_scores = defaultdict(list)

            for sample_i, sample in tqdm(enumerate(sample_generator)):
                # Extract original frame paths for standardized naming
                frame_paths = sample["image_paths"]

                bbox_pipeline.to(accelerator.device)
                best_score = -inf
                if args.if_last_frame_trajectory:
                        sample_bbox = sample['bbox_img'][:args.clip_length]
                        sample_bbox[-1] = sample['bbox_img'][args.clip_length]
                else:
                    sample_bbox = sample['bbox_img']
                for min_guidance_scale, max_guidance_scale in zip([1, 1, 2, 2, 3], [2, 3, 4, 5, 5]):
                    
                    bbox_im = bbox_pipeline(sample['image_init'], 
                                            height=dataset.train_H, width=dataset.train_W, 
                                            bbox_images=sample_bbox.unsqueeze(0),
                                            decode_chunk_size=8, motion_bucket_id=127, fps=args.fps, 
                                            num_inference_steps=30,
                                            num_frames=args.clip_length,
                                            min_guidance_scale=min_guidance_scale,
                                            max_guidance_scale=max_guidance_scale,
                                            noise_aug_strength=args.noise_aug_strength,
                                            generator=generator, output_type='pt',
                                            num_cond_bbox_frames=args.num_cond_bbox_frames).frames[0]
                    bbox_frames = bbox_im.detach().cpu().numpy()*255
                    tmp = bbox_frames.sum(axis=1) < 50
                    bbox_frames[np.repeat(tmp[:, None,::], 3, axis=1)] = 0
                    
                    for frame_i in range(1, bbox_frames.shape[0]-1):
                        if bbox_frames[frame_i].sum(axis=0).min() > 50:
                            # something wrong
                            bbox_frames[frame_i] = np.zeros_like(bbox_frames[frame_i])
                
                    bbox_frames = bbox_frames.astype(np.uint8)
                    clip_miou, clip_ap, clip_ar  = binary_mask_iou(sample['bbox_img_np'][:args.clip_length], bbox_frames)
                    best_score = max(best_score, clip_miou)
                    if best_score == clip_miou:
                        best_generation_bbox = bbox_im
                        best_ap = clip_ap
                        best_ar = clip_ar
                        first_and_last_miou, first_and_last_ap, first_and_last_ar = binary_mask_iou(sample['bbox_img_np'][[0,args.clip_length-1],::], bbox_frames[[0,-1],::])
                        best_generation_np = bbox_frames
                    del bbox_im, bbox_frames, tmp, clip_miou, clip_ar, clip_ap
                
                log_plot = {'inference_step': sample_i}
                log_plot['miou'] = best_score
                log_plot['ap'] = best_ap
                log_plot['ar'] = best_ar
                log_plot['miou_first_last'] = first_and_last_miou
                log_plot['ap_first_last'] = first_and_last_ap
                log_plot['ar_first_last'] = first_and_last_ar
                metric_scores['miou'].append(best_score)
                metric_scores['miou_first_last'].append(first_and_last_miou)
                metric_scores['ap'].append(best_ap)
                metric_scores['ap_first_last'].append(first_and_last_ap)
                metric_scores['ar'].append(best_ar)
                metric_scores['ar_first_last'].append(first_and_last_ar)
                log_plot['avg_miou'] = np.mean(metric_scores['miou'])
                log_plot['std_miou'] = np.std(metric_scores['miou'])
                log_plot['avg_ap'] = np.mean(metric_scores['ap'])
                log_plot['std_ap'] = np.std(metric_scores['ap'])
                log_plot['avg_ar'] = np.mean(metric_scores['ar'])
                log_plot['std_ar'] = np.std(metric_scores['ar'])
                log_plot['avg_miou_first_last'] = np.mean(metric_scores['miou_first_last'])
                log_plot['std_miou_first_last'] = np.std(metric_scores['miou_first_last'])
                log_plot['avg_ap_first_last'] = np.mean(metric_scores['ap_first_last'])
                log_plot['std_ap_first_last'] = np.std(metric_scores['ap_first_last'])
                log_plot['avg_ar_first_last'] = np.mean(metric_scores['ar_first_last'])
                log_plot['std_ar_first_last'] = np.std(metric_scores['ar_first_last'])

                print("Average bbox metrics:", np.mean(metric_scores['miou']), np.mean(metric_scores['ap']), np.mean(metric_scores['ar']), np.mean(metric_scores['miou_first_last']), np.mean(metric_scores['ap_first_last']), np.mean(metric_scores['ar_first_last']))
                print("Std bbox metrics    :", np.std(metric_scores['miou']), np.std(metric_scores['ap']), np.std(metric_scores['ar']), np.std(metric_scores['miou_first_last']), np.std(metric_scores['ap_first_last']), np.std(metric_scores['ar_first_last']))

                # Stage 1 predicted mask (bbox or semantic) - log individual frames with standardized names
                num_frames = min(len(best_generation_np), len(frame_paths), args.clip_length)
                for idx in range(num_frames):
                    orig_name = os.path.basename(frame_paths[idx])
                    
                    if args.use_segmentation:
                        # SEMANTIC MODE
                        new_name = orig_name.replace(".png", "_pred_sem.png")
                    else:
                        # BOUNDING BOX MODE
                        new_name = orig_name.replace(".png", "_predbbox.png")
                    
                    # Transpose from (C, H, W) to (H, W, C) for WandB
                    frame_hwc = np.transpose(best_generation_np[idx], (1, 2, 0))
                    wandb.log({new_name: wandb.Image(frame_hwc)})
                
                bbox_pipeline.to('cpu')

                ctrl_pipeline.to(accelerator.device)
                frames = ctrl_pipeline(sample['image_init'], 
                                    cond_images=2*(best_generation_bbox-0.5).unsqueeze(0),
                                    height=dataset.train_H, width=dataset.train_W, 
                                    decode_chunk_size=8, motion_bucket_id=127, fps=args.fps, 
                                    num_inference_steps=args.num_inference_steps,
                                    num_frames=args.clip_length,
                                    control_condition_scale=args.conditioning_scale,
                                    min_guidance_scale=args.min_guidance_scale,
                                    max_guidance_scale=args.max_guidance_scale,
                                    noise_aug_strength=args.noise_aug_strength,
                                    generator=generator, output_type='pt').frames[0]
                
                ctrl_pipeline.to('cpu')
                accelerator.log(log_plot)
                del log_plot
                           
                frames = frames.detach().cpu().numpy()*255
                frames = frames.astype(np.uint8)
                
                # Stage 2 generated RGB frames - log individual frames with standardized names
                num_frames = min(len(frames), len(frame_paths), args.clip_length)
                for idx in range(num_frames):
                    orig_name = os.path.basename(frame_paths[idx])
                    new_name = orig_name.replace(".png", "_generated.png")
                    # Transpose from (C, H, W) to (H, W, C) for WandB
                    frame_hwc = np.transpose(frames[idx], (1, 2, 0))
                    wandb.log({new_name: wandb.Image(frame_hwc)})
                
                # GT RGB frames - log individual frames with standardized names
                gt_frames = sample["gt_clip_np"]
                num_frames = min(len(gt_frames), len(frame_paths), args.clip_length)
                for idx in range(num_frames):
                    orig_name = os.path.basename(frame_paths[idx])
                    new_name = orig_name.replace(".png", "_gt.png")
                    # Transpose from (C, H, W) to (H, W, C) for WandB
                    frame_hwc = np.transpose(gt_frames[idx], (1, 2, 0))
                    wandb.log({new_name: wandb.Image(frame_hwc)})
                
                # GT segmentation (bbox or semantic) - log individual frames with standardized names
                if args.use_segmentation:
                    # For semantic mode, use segmentation masks if available
                    # Check if segmentation_np exists, otherwise fall back to bbox_img_np
                    if 'segmentation_np' in sample:
                        gt_segs = sample["segmentation_np"]
                    else:
                        gt_segs = sample["bbox_img_np"]
                else:
                    gt_segs = sample["bbox_img_np"]
                
                num_frames = min(len(gt_segs), len(frame_paths), args.clip_length)
                for idx in range(num_frames):
                    orig_name = os.path.basename(frame_paths[idx])
                    
                    if args.use_segmentation:
                        new_name = orig_name.replace(".png", "_gt_sem.png")
                    else:
                        new_name = orig_name.replace(".png", "_gt_bbox.png")
                    
                    # Transpose from (C, H, W) to (H, W, C) for WandB
                    frame_hwc = np.transpose(gt_segs[idx], (1, 2, 0))
                    wandb.log({new_name: wandb.Image(frame_hwc)})
                
                # -----------------------------------------
                # LOG VIDEOS (GIF/MP4) WITH CLEAN NAMING
                # -----------------------------------------
                # Extract scene ID from first frame for video naming
                first_orig_name = os.path.basename(frame_paths[0])
                scene_id = first_orig_name.rsplit("-", 1)[0]  # e.g., 2013_05_28_drive_0000_sync_0000
                
                # Stage-1 predicted mask video (raw tensor → RGB video)
                pred_video = (best_generation_bbox.detach().cpu().numpy() * 255).astype(np.uint8)
                pred_video = np.transpose(pred_video, (0, 2, 3, 1))  # (T, C, H, W) → (T, H, W, C)
                video_key = f"{scene_id}_{sample_i}_stage1_pred_{'sem' if args.use_segmentation else 'bbox'}_video"
                wandb.log({video_key: wandb.Video(pred_video, fps=args.fps)})
                
                # Stage-2 generated RGB video
                wandb.log({f"{scene_id}_{sample_i}_generated_video": wandb.Video(frames, fps=args.fps)})
                
                # GT RGB video
                wandb.log({f"{scene_id}_{sample_i}_gt_video": wandb.Video(gt_frames, fps=args.fps)})
                
                # GT segmentation video (bbox or semantic)
                gt_seg_video = gt_segs[:args.clip_length]
                video_key = f"{scene_id}_{sample_i}_gt_{'sem' if args.use_segmentation else 'bbox'}_video"
                wandb.log({video_key: wandb.Video(gt_seg_video, fps=args.fps)})
                
                del frames, best_generation_bbox, best_generation_np
                del sample
                torch.cuda.empty_cache()
                if sample_i >= args.num_demo_samples:
                    break

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            with torch.autocast(
                str(accelerator.device).replace(":0", ""), enabled=accelerator.mixed_precision == "fp16"
            ):
                from ctrlv.models import UNetSpatioTemporalConditionModel
                if not os.path.isfile(os.path.join(args.pretrained_bbox_model, "unet")):
                    # Get the most recent checkpoint
                    dirs = os.listdir(args.pretrained_bbox_model)
                    dirs = [d for d in dirs if d.startswith("checkpoint")]
                    dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                    path = dirs[-1] if len(dirs) > 0 else None
                    args.pretrained_bbox_model = os.path.join(args.pretrained_bbox_model, path)
                bbox_unet = UNetSpatioTemporalConditionModel.from_pretrained(args.pretrained_bbox_model,
                                                                             subfolder="unet",
                                                                             low_cpu_mem_usage=True, 
                                                                             num_frames=args.clip_length)
                bbox_pipeline = VideoDiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt",
                                                                       unet = bbox_unet,)
                bbox_pipeline = bbox_pipeline.to('cpu')

                from ctrlv.models import UNetSpatioTemporalConditionModel, ControlNetModel
                ctrlnet = ControlNetModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="control_net")
                unet = UNetSpatioTemporalConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
                pipeline = StableVideoControlPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt",
                                                                      controlnet = ctrlnet,
                                                                      unet = unet,)
                pipeline = pipeline.to('cpu')
                result_generator = run_pipelines(bbox_pipeline, pipeline, sample_generator)
        
        logging.info("Finished evaluation.")
        accelerator.end_training()

    except KeyboardInterrupt:
        accelerator.end_training()
        if is_wandb_available():
            wandb.finish()
        print("Keboard interrupt: shutdown requested... Exiting.")
        exit()
    except Exception:
        import sys, traceback
        if is_wandb_available():
            wandb.finish()
        traceback.print_exc(file=sys.stdout)
        sys.exit(0)

if __name__ == '__main__':
    main()