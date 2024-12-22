import torch
import torch.nn.functional as F

def apply_film_grain(frames: torch.Tensor, grain_amount: float = 0.0) -> torch.Tensor:
    """Apply realistic film grain effect to frames.
    
    Args:
        frames: Input tensor of shape [B,C,H,W] in range [0,1]
        grain_amount: Amount of grain to apply (0-100)
        
    Returns:
        Processed frames tensor with film grain
    """
    if grain_amount <= 0:
        return frames
        
    device = frames.device
    B, C, H, W = frames.shape
    
    # Scale grain_amount from 0-100 to reasonable working range
    grain_intensity = grain_amount / 400.0  # Scaled down to prevent overwhelming effect
    
    # Generate luminance noise pattern
    noise = torch.randn((B, 1, H, W), device=device) * grain_intensity
    
    # Apply threshold and dilation-like effects to create grain clusters
    noise = torch.clamp(noise, -2 * grain_intensity, 2 * grain_intensity)
    noise = F.avg_pool2d(noise, kernel_size=3, stride=1, padding=1)
    
    # Convert frames to YCbCr-like space for luminance-based processing
    # Approximate conversion weights from RGB to Y
    y = frames[:, 0] * 0.299 + frames[:, 1] * 0.587 + frames[:, 2] * 0.114
    y = y.unsqueeze(1)
    
    # Create luminance-dependent grain mask
    # Reduce grain in very dark and very bright areas
    lum_mask = 1.0 - (2.0 * torch.abs(y - 0.5))
    lum_mask = torch.clamp(lum_mask, 0.2, 1.0)
    
    # Modulate noise by luminance mask
    noise = noise * lum_mask
    
    # Apply grain primarily to luminance while preserving color
    graininess = 0.85  # Proportion of grain to apply to luminance vs color
    
    # Apply to luminance
    frames_out = frames.clone()
    frames_out = frames_out + noise * graininess
    
    # Apply small amount to color channels to maintain some color noise
    color_noise = torch.randn_like(frames) * (grain_intensity * (1 - graininess) * 0.5)
    frames_out = frames_out + color_noise
    
    # Final contrast adjustment
    frames_out = torch.clamp(frames_out, 0.0, 1.0)
    contrast = 1.0 + (grain_amount / 200.0)  # Subtle contrast boost based on grain amount
    frames_out = (frames_out - 0.5) * contrast + 0.5
    frames_out = torch.clamp(frames_out, 0.0, 1.0)
    
    return frames_out