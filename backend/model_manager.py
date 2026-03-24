"""
Model Manager: Loads and manages Stage 1 and Stage 2 pipelines for inference.
"""

import os
import sys
import logging
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Optional, List, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange

logger = logging.getLogger(__name__)

# Ensure ctrlv is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Paths
SEMANTIC_VAE_CHECKPOINT = '/usrhomes/s1492/vae_semantic/checkpoints/semantic_vae_native/best_model_with_dice_boundaryweight.pth'
BASE_MODEL = 'stabilityai/stable-video-diffusion-img2vid-xt'
STAGE1_CHECKPOINT_DIR = '/no_backups/s1492/Ctrl-V/checkpoints/kitti360_semantic_predict_vae'
STAGE2_CHECKPOINT_DIR = '/no_backups/s1492/Ctrl-V/checkpoints/kitti360_sem2video_unet_unfreeze_reinject'
# STAGE2_CHECKPOINT_DIR = '/no_backups/s1492/Ctrl-V/checkpoints/kitti360_semantic2video_vae'
DRN_DIR = '/usrhomes/s1492/drn'
DRN_CHECKPOINT = '/usrhomes/s1492/drn/KITTI360_checkpoints/checkpoint_030.pth.tar'
DRN_INFO_JSON = '/usrhomes/s1492/drn/CTRLV_BBOX/info.json'
KITTI360_ROOT = '/misc/data/public/kitti-360/KITTI-360'


def find_latest_checkpoint(checkpoint_dir: str) -> Tuple[str, int]:
    """Find latest checkpoint-XXXXX in a directory."""
    subdirs = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint")]
    if not subdirs:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")
    subdirs = sorted(subdirs, key=lambda x: int(x.split("-")[1]))
    latest = subdirs[-1]
    step = int(latest.split("-")[1])
    return os.path.join(checkpoint_dir, latest), step


class ModelManager:
    """Manages model loading and inference for both stages."""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weight_dtype = torch.float16

        self.stage1_pipeline = None
        self.stage2_pipeline = None
        self.vae_manager = None
        self.drn_model = None
        self.dataset = None
        self.dataloader = None

        self.stage1_step = None
        self.stage2_step = None

    def _load_vae_manager(self, vae):
        """Load DualVAEManager if not already loaded."""
        if self.vae_manager is not None:
            return self.vae_manager

        from ctrlv.models import DualVAEManager
        self.vae_manager = DualVAEManager(
            rgb_vae=vae,
            semantic_vae_checkpoint=SEMANTIC_VAE_CHECKPOINT,
            num_semantic_classes=19,
            device=self.device,
            clip_size=25,
            verbose=True
        )
        return self.vae_manager

    def load_stage1(self, log_fn=None):
        """Load Stage 1 pipeline (RGB -> Semantic)."""
        if self.stage1_pipeline is not None:
            return

        def log(msg):
            logger.info(msg)
            if log_fn:
                log_fn(msg)

        log("Loading Stage 1 pipeline...")
        ckpt_path = os.path.join(STAGE1_CHECKPOINT_DIR, 'best_checkpoint')
        self.stage1_step = 'best'
        log(f"  Checkpoint: {ckpt_path} (step {self.stage1_step})")

        from diffusers import EulerDiscreteScheduler
        from diffusers.models import AutoencoderKLTemporalDecoder
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
        from ctrlv.models import UNetSpatioTemporalConditionModel
        from ctrlv.pipelines import VideoDiffusionPipeline

        vae = AutoencoderKLTemporalDecoder.from_pretrained(
            BASE_MODEL, subfolder="vae", variant="fp16"
        )
        unet = UNetSpatioTemporalConditionModel.from_pretrained(
            BASE_MODEL, subfolder="unet", variant="fp16",
            low_cpu_mem_usage=True, num_frames=25
        )
        feature_extractor = CLIPImageProcessor.from_pretrained(
            BASE_MODEL, subfolder="feature_extractor"
        )
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            BASE_MODEL, subfolder="image_encoder", variant="fp16"
        )

        log("  Loading UNet checkpoint weights...")
        unet_ckpt_path = os.path.join(ckpt_path, "unet")
        if os.path.exists(unet_ckpt_path):
            load_model = UNetSpatioTemporalConditionModel.from_pretrained(
                ckpt_path, subfolder="unet"
            )
            unet.register_to_config(**load_model.config)
            unet.load_state_dict(load_model.state_dict())
            del load_model

        vae_manager = self._load_vae_manager(vae)

        vae.to(self.device, dtype=self.weight_dtype)
        unet.to(self.device, dtype=self.weight_dtype)
        image_encoder.to(self.device, dtype=self.weight_dtype)
        unet.eval()

        pipeline = VideoDiffusionPipeline.from_pretrained(
            BASE_MODEL,
            unet=unet, image_encoder=image_encoder, vae=vae,
            feature_extractor=feature_extractor,
            variant="fp16", torch_dtype=self.weight_dtype,
        )
        pipeline = pipeline.to(self.device)
        pipeline.set_progress_bar_config(disable=True)
        pipeline.vae_manager = vae_manager

        self.stage1_pipeline = pipeline
        log("  Stage 1 pipeline loaded successfully")

    def load_stage2(self, log_fn=None):
        """Load Stage 2 pipeline (Semantic -> RGB)."""
        if self.stage2_pipeline is not None:
            return

        def log(msg):
            logger.info(msg)
            if log_fn:
                log_fn(msg)

        log("Loading Stage 2 pipeline...")
        best_ckpt = os.path.join(STAGE2_CHECKPOINT_DIR, 'best_checkpoint')
        if os.path.exists(best_ckpt):
            ckpt_path = best_ckpt
            self.stage2_step = 'best'
        else:
            ckpt_path, self.stage2_step = find_latest_checkpoint(STAGE2_CHECKPOINT_DIR)
        log(f"  Checkpoint: {ckpt_path} (step {self.stage2_step})")

        from diffusers.models import AutoencoderKLTemporalDecoder
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
        from ctrlv.models import UNetSpatioTemporalConditionModel, ControlNetModel
        from ctrlv.pipelines import StableVideoControlPipeline

        ctrlnet = ControlNetModel.from_pretrained(ckpt_path, subfolder="control_net")
        log("  ControlNet loaded")

        unet = UNetSpatioTemporalConditionModel.from_pretrained(
            ckpt_path, subfolder="unet",
            low_cpu_mem_usage=True, num_frames=25
        )
        log("  UNet loaded")

        vae = AutoencoderKLTemporalDecoder.from_pretrained(
            BASE_MODEL, subfolder="vae", variant="fp16"
        )
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            BASE_MODEL, subfolder="image_encoder", variant="fp16"
        )
        feature_extractor = CLIPImageProcessor.from_pretrained(
            BASE_MODEL, subfolder="feature_extractor"
        )

        vae_manager = self._load_vae_manager(vae)

        vae.to(self.device, dtype=self.weight_dtype)
        unet.to(self.device, dtype=self.weight_dtype)
        image_encoder.to(self.device, dtype=self.weight_dtype)
        ctrlnet.to(self.device, dtype=self.weight_dtype)
        ctrlnet.eval()
        unet.eval()

        pipeline = StableVideoControlPipeline.from_pretrained(
            BASE_MODEL,
            unet=unet, controlnet=ctrlnet,
            image_encoder=image_encoder, vae=vae,
            feature_extractor=feature_extractor,
            variant="fp16", torch_dtype=self.weight_dtype,
        )
        pipeline = pipeline.to(self.device)
        pipeline.set_progress_bar_config(disable=True)
        pipeline.vae_manager = vae_manager

        self.stage2_pipeline = pipeline
        log("  Stage 2 pipeline loaded successfully")

    def load_dataset(self, log_fn=None):
        """Load validation dataset."""
        if self.dataset is not None:
            return

        def log(msg):
            logger.info(msg)
            if log_fn:
                log_fn(msg)

        log("Loading KITTI-360 validation dataset...")
        from ctrlv.utils import get_dataloader

        self.dataset, self.dataloader = get_dataloader(
            '', 'kitti360', if_train=False,
            clip_length=25, batch_size=1, num_workers=2,
            data_type='clip', use_default_collate=True, tokenizer=None, shuffle=False,
            if_return_bbox_im=True, train_H=192, train_W=704,
            use_segmentation=True, use_preplotted_bbox=True,
            if_last_frame_traj=False, non_overlapping_clips=True,
            return_semantic_ids=True
        )
        log(f"  Dataset loaded: {len(self.dataset)} clips")

    def get_dataset_samples(self, count=20):
        """Get sample thumbnails from the dataset."""
        self.load_dataset()
        from ctrlv.utils import get_n_training_samples
        samples = get_n_training_samples(self.dataloader, min(count, len(self.dataset)), show_progress=False)
        return samples

    def run_stage1_inference(
        self,
        image_init,
        semantic_ids: torch.Tensor,
        bbox_img: torch.Tensor,
        num_frames: int = 25,
        num_inference_steps: int = 30,
        min_guidance_scale: float = 3.0,
        max_guidance_scale: float = 7.0,
        noise_aug_strength: float = 0.01,
        fps: int = 7,
        seed: int = 1234,
        num_cond_bbox_frames: int = 1,
        log_fn=None,
    ):
        """
        Run Stage 1 inference (RGB -> Semantic).

        Args:
            image_init: PIL Image or tensor for CLIP conditioning
            semantic_ids: [T, H, W] int64 trainIDs (0-18) for semantic VAE conditioning
            bbox_img: [T, 3, H, W] float32 RGB semantic visualization
            Returns: pred_semantic_ids [T, H, W] numpy array of trainIDs
        """
        def log(msg):
            logger.info(msg)
            if log_fn:
                log_fn(msg)

        self.load_stage1(log_fn)

        log("Running Stage 1 inference...")
        generator = torch.Generator(device=self.device).manual_seed(seed)

        bbox_img_rgb = bbox_img.unsqueeze(0)  # [1, T, 3, H, W]
        semantic_ids_cond = semantic_ids.unsqueeze(0)  # [1, T, H, W]

        log(f"  Inference params: steps={num_inference_steps}, guidance=[{min_guidance_scale}, {max_guidance_scale}]")
        log(f"  Input: image_init={type(image_init)}, semantic_ids={semantic_ids_cond.shape}")

        with torch.autocast(str(self.device).replace(":0", ""), enabled=True):
            result = self.stage1_pipeline(
                image_init,
                height=192, width=704,
                bbox_images=bbox_img_rgb,
                decode_chunk_size=8, motion_bucket_id=127, fps=fps,
                num_inference_steps=num_inference_steps,
                num_frames=num_frames,
                min_guidance_scale=min_guidance_scale,
                max_guidance_scale=max_guidance_scale,
                noise_aug_strength=noise_aug_strength,
                generator=generator,
                output_type='latent',
                num_cond_bbox_frames=num_cond_bbox_frames,
                semantic_ids=semantic_ids_cond,
                use_semantic_vae=True,
            )

        log("  Decoding semantic latents...")
        latents = result.frames[0].to(torch.float32)
        pred_semantic_ids = self.vae_manager.decode_semantic(latents)
        pred_np = pred_semantic_ids.cpu().numpy()

        log(f"  Generated semantic frames: {pred_np.shape}")
        return pred_np

    def run_stage2_inference(
        self,
        image_init,
        semantic_ids: torch.Tensor,
        bbox_img: torch.Tensor,
        num_frames: int = 25,
        num_inference_steps: int = 30,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 3.0,
        conditioning_scale: float = 1.0,
        noise_aug_strength: float = 0.01,
        fps: int = 7,
        seed: int = 1234,
        log_fn=None,
    ):
        """
        Run Stage 2 inference (Semantic -> RGB).

        Args:
            image_init: PIL Image for CLIP conditioning
            semantic_ids: [T, H, W] int64 trainIDs for ControlNet conditioning
            bbox_img: [T, 3, H, W] float32 RGB semantic visualization
            Returns: gen_frames [T, H, W, 3] uint8 numpy array
        """
        def log(msg):
            logger.info(msg)
            if log_fn:
                log_fn(msg)

        self.load_stage2(log_fn)

        log("Running Stage 2 inference...")
        generator = torch.Generator(device=self.device).manual_seed(seed)

        bbox_img_rgb = bbox_img.unsqueeze(0)  # [1, T, 3, H, W]
        semantic_ids_cond = semantic_ids.unsqueeze(0)  # [1, T, H, W]

        log(f"  Inference params: steps={num_inference_steps}, guidance=[{min_guidance_scale}, {max_guidance_scale}], cond_scale={conditioning_scale}")

        with torch.autocast(str(self.device).replace(":0", ""), enabled=True):
            result = self.stage2_pipeline(
                image_init,
                cond_images=bbox_img_rgb,
                height=192, width=704,
                decode_chunk_size=8, motion_bucket_id=127, fps=fps,
                num_inference_steps=num_inference_steps,
                num_frames=num_frames,
                control_condition_scale=conditioning_scale,
                min_guidance_scale=min_guidance_scale,
                max_guidance_scale=max_guidance_scale,
                noise_aug_strength=noise_aug_strength,
                generator=generator,
                output_type='pt',
                semantic_ids=semantic_ids_cond,
                use_semantic_vae=True,
            )

        log("  Processing generated frames...")
        gen_frames_pt = result.frames[0]  # [T, 3, H, W]
        gen_frames_np = (gen_frames_pt.detach().cpu().numpy() * 255).astype(np.uint8)
        gen_frames_hwc = np.transpose(gen_frames_np, (0, 2, 3, 1))  # [T, H, W, 3]

        log(f"  Generated RGB frames: {gen_frames_hwc.shape}")
        return gen_frames_hwc

    def load_drn(self, log_fn=None):
        """Load DRN segmentation model for Stage 2 evaluation."""
        if self.drn_model is not None:
            return

        def log(msg):
            logger.info(msg)
            if log_fn:
                log_fn(msg)

        log("Loading DRN model...")
        sys.path.insert(0, DRN_DIR)
        import drn as drn_module
        from segment import DRNSeg

        model = DRNSeg('drn_d_105', 19, pretrained_model=None, pretrained=False)
        state_dict = torch.load(DRN_CHECKPOINT, map_location='cpu')["state_dict"]
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        model = model.cuda()
        model.eval()
        self.drn_model = model
        log("  DRN model loaded")

    def evaluate_stage1(self, pred_semantic_ids: np.ndarray, gt_semantic_ids: np.ndarray):
        """
        Evaluate Stage 1: compare predicted vs GT semantic IDs.
        Returns metrics dict.
        """
        from ctrlv.utils.semantic_preprocessing import KITTI360_CLASS_NAMES

        num_classes = 19
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

        pred = pred_semantic_ids
        gt = gt_semantic_ids

        # Resize if needed
        if pred.shape != gt.shape:
            pred_t = torch.from_numpy(pred).unsqueeze(1).float()
            pred_t = F.interpolate(pred_t, size=gt.shape[1:], mode='nearest')
            pred = pred_t.squeeze(1).numpy().astype(np.int64)

        for t in range(pred.shape[0]):
            valid_mask = (gt[t] != 255) & (gt[t] < num_classes)
            p = pred[t][valid_mask]
            g = gt[t][valid_mask]
            for gi, pi in zip(g.flatten(), p.flatten()):
                confusion_matrix[gi, pi] += 1

        tp = np.diag(confusion_matrix)
        fp = confusion_matrix.sum(axis=0) - tp
        fn = confusion_matrix.sum(axis=1) - tp
        gt_per_class = confusion_matrix.sum(axis=1)

        denominator = tp + fp + fn
        iou_per_class = np.where(denominator > 0, tp / denominator, np.nan)
        acc_per_class = np.where(gt_per_class > 0, tp / gt_per_class, np.nan)

        miou = float(np.nanmean(iou_per_class))
        overall_accuracy = float(tp.sum() / gt_per_class.sum()) if gt_per_class.sum() > 0 else 0.0
        mean_accuracy = float(np.nanmean(acc_per_class))
        freq = gt_per_class / gt_per_class.sum() if gt_per_class.sum() > 0 else np.zeros(num_classes)
        fwiou = float(np.nansum(freq * iou_per_class))

        per_class = {}
        for i, name in enumerate(KITTI360_CLASS_NAMES):
            per_class[name] = {
                'iou': None if np.isnan(iou_per_class[i]) else float(iou_per_class[i]),
                'accuracy': None if np.isnan(acc_per_class[i]) else float(acc_per_class[i]),
            }

        return {
            'miou': miou,
            'overall_accuracy': overall_accuracy,
            'mean_accuracy': mean_accuracy,
            'fwiou': fwiou,
            'per_class': per_class,
        }

    def evaluate_stage2_drn(self, gen_frames_hwc: np.ndarray, gt_semantic_ids: np.ndarray, log_fn=None):
        """
        Evaluate Stage 2: Run DRN on generated RGB, compare with GT semantic.
        Returns metrics dict.
        """
        import json as json_module
        from torchvision import transforms as T
        from ctrlv.utils.semantic_preprocessing import KITTI360_CLASS_NAMES

        def log(msg):
            logger.info(msg)
            if log_fn:
                log_fn(msg)

        self.load_drn(log_fn)

        with open(DRN_INFO_JSON) as f:
            info = json_module.load(f)

        normalize = T.Normalize(mean=info['mean'], std=info['std'])
        transform = T.Compose([T.ToTensor(), normalize])

        log("Running DRN on generated frames...")
        drn_preds = []
        for t in range(gen_frames_hwc.shape[0]):
            img_pil = Image.fromarray(gen_frames_hwc[t])
            img_tensor = transform(img_pil).unsqueeze(0).cuda()
            with torch.no_grad():
                output = self.drn_model(img_tensor)[0]
            _, pred = torch.max(output, 1)
            drn_preds.append(pred.squeeze().cpu().numpy())
        drn_pred = np.stack(drn_preds, axis=0)

        # Resize if needed
        if drn_pred.shape[1:] != gt_semantic_ids.shape[1:]:
            drn_pred_t = torch.from_numpy(drn_pred).unsqueeze(1).float()
            drn_pred_t = F.interpolate(drn_pred_t, size=gt_semantic_ids.shape[1:], mode='nearest')
            drn_pred = drn_pred_t.squeeze(1).numpy().astype(np.int64)

        log("Computing DRN metrics...")
        return self.evaluate_stage1(drn_pred, gt_semantic_ids)

    def compute_image_metrics(self, gt_frames_hwc: np.ndarray, gen_frames_hwc: np.ndarray, log_fn=None):
        """Compute LPIPS, SSIM, PSNR between GT and generated frames."""
        def log(msg):
            logger.info(msg)
            if log_fn:
                log_fn(msg)

        results = {}

        # SSIM & PSNR
        try:
            from skimage.metrics import structural_similarity as ssim_fn
            from skimage.metrics import peak_signal_noise_ratio as psnr_fn

            ssim_vals = []
            psnr_vals = []
            T = min(gt_frames_hwc.shape[0], gen_frames_hwc.shape[0])
            for t in range(T):
                gt_f = gt_frames_hwc[t].astype(np.float32) / 255.0
                gen_f = gen_frames_hwc[t].astype(np.float32) / 255.0
                ssim_val = ssim_fn(gt_f, gen_f, channel_axis=2, data_range=1.0)
                psnr_val = psnr_fn(gt_f, gen_f, data_range=1.0)
                ssim_vals.append(ssim_val)
                psnr_vals.append(psnr_val)

            results['ssim'] = float(np.mean(ssim_vals))
            results['psnr'] = float(np.mean(psnr_vals))
            log(f"  SSIM: {results['ssim']:.4f}, PSNR: {results['psnr']:.4f}")
        except ImportError:
            log("  skimage not available, skipping SSIM/PSNR")

        # LPIPS
        try:
            import lpips
            loss_fn = lpips.LPIPS(net='alex').cuda()
            T = min(gt_frames_hwc.shape[0], gen_frames_hwc.shape[0])

            lpips_vals = []
            for t in range(T):
                gt_t = torch.from_numpy(gt_frames_hwc[t]).permute(2, 0, 1).float() / 127.5 - 1.0
                gen_t = torch.from_numpy(gen_frames_hwc[t]).permute(2, 0, 1).float() / 127.5 - 1.0
                with torch.no_grad():
                    val = loss_fn(gt_t.unsqueeze(0).cuda(), gen_t.unsqueeze(0).cuda())
                lpips_vals.append(val.item())

            results['lpips'] = float(np.mean(lpips_vals))
            del loss_fn
            log(f"  LPIPS: {results['lpips']:.4f}")
        except ImportError:
            log("  lpips not available, skipping")

        return results
