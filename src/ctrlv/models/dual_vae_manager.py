"""
Dual VAE Manager for Ctrl-V with Semantic VAE Integration

This module manages both RGB VAE (for video frames) and Semantic VAE (for semantic maps).

IMPORTANT: Semantic images are GRAYSCALE labels (not RGB), which are:
1. Loaded as grayscale PNG
2. Remapped to trainIds (0-18) using KITTI360_LABEL_MAPPING
3. Converted to one-hot encoding
4. Processed by Semantic VAE

It provides a unified interface for encoding both types of inputs during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict
import numpy as np

# Add semantic VAE to path
sys.path.insert(0, '/usrhomes/s1492/vae_semantic')
sys.path.insert(0, '/usrhomes/s1492/Ctrl-V-seg/src')

try:
    from semantic_vae_native_inference import SemanticVAEInference
except ImportError:
    print("Warning: Could not import SemanticVAEInference. Make sure /usrhomes/s1492/vae_semantic is accessible.")
    SemanticVAEInference = None

try:
    from ctrlv.utils.semantic_preprocessing import semantic_ids_to_onehot
except ImportError:
    print("Warning: Could not import semantic_preprocessing utilities.")
    semantic_ids_to_onehot = None


class DualVAEManager(nn.Module):
    """
    Manages both RGB VAE and Semantic VAE for Ctrl-V training.
    
    - RGB VAE: Encodes/decodes RGB video frames (frozen, from SVD)
    - Semantic VAE: Encodes/decodes semantic ID maps (frozen, pre-trained)
    
    Usage:
        vae_manager = DualVAEManager(
            rgb_vae=pretrained_rgb_vae,
            semantic_vae_checkpoint='/path/to/semantic_vae.pth',
            device='cuda'
        )
        
        # Encode RGB frames
        rgb_latents = vae_manager.encode_rgb(rgb_frames)
        
        # Encode semantic maps (from semantic IDs)
        semantic_latents = vae_manager.encode_semantic(semantic_ids)
    """
    
    def __init__(
        self,
        rgb_vae: nn.Module,
        semantic_vae_checkpoint: str,
        num_semantic_classes: int = 19,
        device: str = 'cuda',
        clip_size: int = 4,  # Temporal frames for semantic VAE
        verbose: bool = True
    ):
        """
        Initialize dual VAE manager.
        
        Args:
            rgb_vae: Pre-trained RGB VAE (AutoencoderKLTemporalDecoder)
            semantic_vae_checkpoint: Path to trained semantic VAE checkpoint
            num_semantic_classes: Number of semantic classes (19 for KITTI-360)
            device: Device to run on
            clip_size: Temporal clip size for semantic VAE (must match training)
            verbose: Print initialization info
        """
        super().__init__()
        
        self.device = device
        self.num_semantic_classes = num_semantic_classes
        self.clip_size = clip_size
        self.verbose = verbose
        
        # Store RGB VAE (already loaded by training script)
        self.rgb_vae = rgb_vae
        self.rgb_vae.requires_grad_(False)
        self.rgb_vae.eval()
        
        # Load Semantic VAE
        if SemanticVAEInference is None:
            raise ImportError(
                "SemanticVAEInference not available. "
                "Make sure /usrhomes/s1492/vae_semantic is in your Python path."
            )
        
        self.semantic_vae = SemanticVAEInference(
            checkpoint_path=semantic_vae_checkpoint,
            device=device,
            num_classes=num_semantic_classes,
            verbose=verbose
        )
        
        # Freeze semantic VAE
        self.semantic_vae.model.requires_grad_(False)
        self.semantic_vae.model.eval()
        
        if verbose:
            print(f"✓ DualVAEManager initialized")
            print(f"  RGB VAE: {type(rgb_vae).__name__}")
            print(f"  Semantic VAE: Loaded from {Path(semantic_vae_checkpoint).name}")
            print(f"  Semantic classes: {num_semantic_classes}")
            print(f"  Clip size: {clip_size}")
    
    def encode_rgb(self, rgb_frames: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB frames using RGB VAE.
        
        Args:
            rgb_frames: RGB frames [B, C, H, W] or [B*F, C, H, W]
        
        Returns:
            latents: Encoded latents
        """
        with torch.no_grad():
            # Ensure consistent dtype with VAE
            original_dtype = rgb_frames.dtype
            if self.rgb_vae.dtype != rgb_frames.dtype:
                rgb_frames = rgb_frames.to(self.rgb_vae.dtype)
            latents = self.rgb_vae.encode(rgb_frames).latent_dist.sample()
            # Convert back if needed
            if original_dtype != latents.dtype:
                latents = latents.to(original_dtype)
        return latents
    
    def encode_semantic_from_grayscale(
        self,
        semantic_grayscale: torch.Tensor
    ) -> torch.Tensor:
        """
        Use encode_semantic_from_ids directly.
        
        The semantic images are already preprocessed grayscale labels with trainIds (0-18).
        This method is kept for backward compatibility but just calls encode_semantic_from_ids.
        
        Args:
            semantic_grayscale: Already remapped semantic IDs [B*F, H, W]
        
        Returns:
            latents: Encoded semantic latents [B*F, C, H_latent, W_latent]
        """
        return self.encode_semantic_from_ids(semantic_grayscale)
    
    def encode_semantic_from_ids(
        self,
        semantic_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode semantic ID maps using Semantic VAE.
        
        IMPORTANT: semantic_ids should be remapped trainIds (0-18), NOT raw KITTI-360 IDs.
        The preprocessing (grayscale -> trainIds) should be done in the dataset.
        
        Args:
            semantic_ids: Semantic trainIds [B*F, H, W] with values in [0, 18] or 255 (ignore)
        
        Returns:
            latents: Encoded semantic latents [B*F, C, H_latent, W_latent]
        """
        bf, h, w = semantic_ids.shape
        device = semantic_ids.device
        
        # Process in batches (semantic VAE expects temporal dimension)
        # Reshape to [B, T, H, W] where T=clip_size
        if bf % self.clip_size == 0:
            batch_size = bf // self.clip_size
            semantic_ids = semantic_ids.view(batch_size, self.clip_size, h, w)
            need_unpad = False
        else:
            # Pad to make divisible by clip_size
            pad_size = self.clip_size - (bf % self.clip_size)
            semantic_ids = torch.cat([
                semantic_ids,
                semantic_ids[-1:].repeat(pad_size, 1, 1)
            ], dim=0)
            bf_padded = semantic_ids.shape[0]
            batch_size = bf_padded // self.clip_size
            semantic_ids = semantic_ids.view(batch_size, self.clip_size, h, w)
            need_unpad = True
        
        # Encode through semantic VAE
        # The semantic VAE model internally converts IDs -> one-hot -> features -> latents
        with torch.no_grad():
            # semantic_vae.model.forward() expects [B, T, H, W] trainIds
            # Returns logits [B, T, 19, H, W], but we want to use encode() for latents only
            
            # Use the model's internal encode path
            B, T, H_in, W_in = semantic_ids.shape
            
            # Step 1: One-hot encoding (same as model does internally)
            semantic_flat = semantic_ids.view(B * T, H_in, W_in)
            semantic_clamped = torch.clamp(semantic_flat, 0, self.num_semantic_classes - 1)
            x_onehot = F.one_hot(semantic_clamped.long(), num_classes=self.num_semantic_classes)
            x_onehot = x_onehot.permute(0, 3, 1, 2).float()  # [B*T, 19, H, W]
            
            # Step 2: Semantic stem
            h0 = self.semantic_vae.model.semantic_stem(x_onehot)  # [B*T, 128, H, W]
            
            # Step 3: VAE encoder core
            latents_flat = self.semantic_vae.model._encode_semantic_features(h0)  # [B*T, 4, H/8, W/8]
            
            # Reshape to [B, T, C, H_latent, W_latent]
            _, C, H_latent, W_latent = latents_flat.shape
            latents = latents_flat.view(B, T, C, H_latent, W_latent)
            
            # Flatten temporal dimension: [B, T, C, H, W] -> [B*T, C, H, W]
            latents = latents.view(-1, C, H_latent, W_latent)
            
            # Remove padding if added
            if need_unpad:
                latents = latents[:bf]
        
        return latents
    
    
    def decode_rgb(self, latents: torch.Tensor, **kwargs) -> torch.Tensor:
        """Decode latents to RGB frames using RGB VAE."""
        with torch.no_grad():
            return self.rgb_vae.decode(latents, **kwargs)
    
    def decode_semantic(self, latents: torch.Tensor, unscale: bool = True) -> torch.Tensor:
        """
        Decode latents to semantic IDs using Semantic VAE.

        IMPORTANT: Diffusion training scales latents by vae.config.scaling_factor (0.18215).
        The semantic VAE decoder expects UNSCALED latents (raw encoder mean z).
        When decoding diffusion outputs, we must divide by scaling_factor first.

        Args:
            latents: [B*F, C, H_latent, W_latent] (already flat from encode)
            unscale: If True (default), divide by RGB VAE scaling_factor before decoding.
                     Set to False only if latents are already unscaled.

        Returns:
            semantic_ids: [B*F, H, W] trainIDs (0-18)
        """
        with torch.no_grad():
            # Mirror the encode structure:
            # Encode: semantic_stem -> _encode_semantic_features -> latents (unscaled)
            # Training: latents * scaling_factor -> diffusion target (scaled)
            # Decode: latents / scaling_factor -> _decode_to_semantic_features -> semantic_head -> logits

            bf = latents.shape[0]

            # Unscale latents from diffusion space back to VAE latent space
            # The RGB VAE's decode_latents() does this same unscaling internally,
            # but since we bypass that path with output_type='latent', we must do it here.
            if unscale:
                scaling_factor = self.rgb_vae.config.scaling_factor  # 0.18215
                latents = latents / scaling_factor

            # Decode latents to semantic features [B*F, 128, H, W]
            decoded_features = self.semantic_vae.model._decode_to_semantic_features(latents)
            
            # semantic_head expects [B, T, 128, H, W], but we have [B*F, 128, H, W]
            # Reshape to [B, 1, 128, H, W] for single-frame processing
            decoded_features = decoded_features.unsqueeze(1)  # [B*F, 1, 128, H, W]
            
            # Apply semantic head to get class logits [B*F, 1, 19, H, W]
            logits = self.semantic_vae.model.semantic_head(decoded_features)
            
            # Remove temporal dimension and get predictions [B*F, H, W]
            logits = logits[:, 0, :, :, :]  # [B*F, 19, H, W]
            semantic_ids = torch.argmax(logits, dim=1)  # [B*F, H, W]
        
        return semantic_ids
    
    @property
    def config(self):
        """Return RGB VAE config for compatibility."""
        return self.rgb_vae.config
    
    @property
    def dtype(self):
        """Return RGB VAE dtype."""
        return self.rgb_vae.dtype
    
    def to(self, *args, **kwargs):
        """Move both VAEs to device/dtype."""
        self.rgb_vae = self.rgb_vae.to(*args, **kwargs)
        # Semantic VAE model is already on correct device from initialization
        return self
    
    def eval(self):
        """Set both VAEs to eval mode."""
        self.rgb_vae.eval()
        self.semantic_vae.model.eval()
        return self
    
    def train(self, mode=True):
        """Keep VAEs in eval mode (frozen)."""
        return self.eval()
