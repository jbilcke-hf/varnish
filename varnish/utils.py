import math
from typing import Union, List
import torch
import os
import io
import base64
from datetime import datetime
import numpy as np
import itertools
from PIL import Image
import safetensors.torch
import tqdm
import logging
from spandrel import ModelLoader

logger = logging.getLogger(__file__)

def is_truthy(val):
    """Check if a value is truthy"""
    if not val:
        return False
    return str(val).lower() in ('true', '1', 't', 'y', 'yes')

def print_directory_structure(startpath):
    """Print the directory structure starting from the given path."""
    logger.info(f'💡 Printing directory structure of "{startpath}":')
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * level
        logger.info(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            logger.info(f"{subindent}{f}")

def process_input_image(image_data: str, target_width: int, target_height: int, input_image_quality: int) -> Image.Image:
    """
    Process input image from base64, resize and crop to target dimensions
    
    Args:
        image_data: Base64 encoded image data
        target_width: Desired width
        target_height: Desired height
        fake_video_input: bool
        
    Returns:
        Processed PIL Image
    """
    try:
        # Handle data URI format
        if image_data.startswith('data:'):
            image_data = image_data.split(',', 1)[1]
            
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode not in ('RGB', 'RGBA'):
            image = image.convert('RGB')
        elif image.mode == 'RGBA':
            # Handle transparency by compositing on white background
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
            
        # Calculate target aspect ratio
        target_aspect = target_width / target_height
        
        # Get current dimensions
        orig_width, orig_height = image.size
        orig_aspect = orig_width / orig_height
        
        # Calculate dimensions for resizing
        if orig_aspect > target_aspect:
            # Image is wider than target
            new_height = target_height
            new_width = int(target_height * orig_aspect)
        else:
            # Image is taller than target
            new_width = target_width
            new_height = int(target_width / orig_aspect)
            
        # Resize image
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Center crop to target dimensions
        left = (new_width - target_width) // 2
        top = (new_height - target_height) // 2
        right = left + target_width
        bottom = top + target_height
        
        image = image.crop((left, top, right, bottom))

        # Apply JPEG compression if input_image_quality is not 100
        if input_image_quality < 100:
            # Save with compression to bytes buffer
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=input_image_quality)
            buffer.seek(0)
            # Load compressed image back
            image = Image.open(buffer)
        
        return image
        
    except Exception as e:
        raise ValueError(f"Failed to process input image: {str(e)}")

def load_torch_file(ckpt, device=None, dtype=torch.float16):
    if device is None:
        device = torch.device("cpu")
    if ckpt.lower().endswith(".safetensors") or ckpt.lower().endswith(".sft"):
        sd = safetensors.torch.load_file(ckpt, device=device.type)
    else:
        if not "weights_only" in torch.load.__code__.co_varnames:
            logger.warning(
                "Warning torch.load doesn't support weights_only on this pytorch version, loading unsafely."
            )

        pl_sd = torch.load(ckpt, map_location=device, weights_only=True)
        if "global_step" in pl_sd:
            logger.debug(f"Global Step: {pl_sd['global_step']}")
        if "state_dict" in pl_sd:
            sd = pl_sd["state_dict"]
        elif "params_ema" in pl_sd:
            sd = pl_sd["params_ema"]
        else:
            sd = pl_sd

    sd = {k: v.to(dtype) for k, v in sd.items()}
    return sd


def state_dict_prefix_replace(state_dict, replace_prefix, filter_keys=False):
    if filter_keys:
        out = {}
    else:
        out = state_dict
    for rp in replace_prefix:
        replace = list(
            map(
                lambda a: (a, "{}{}".format(replace_prefix[rp], a[len(rp) :])),
                filter(lambda a: a.startswith(rp), state_dict.keys()),
            )
        )
        for x in replace:
            w = state_dict.pop(x[0])
            out[x[1]] = w
    return out


def module_size(module):
    module_mem = 0
    sd = module.state_dict()
    for k in sd:
        t = sd[k]
        module_mem += t.nelement() * t.element_size()
    return module_mem


def get_tiled_scale_steps(width, height, tile_x, tile_y, overlap):
    return math.ceil((height / (tile_y - overlap))) * math.ceil((width / (tile_x - overlap)))


@torch.inference_mode()
def tiled_scale_multidim(
    samples, function, tile=(64, 64), overlap=8, upscale_amount=4, out_channels=3, output_device="cpu", pbar=None
):
    dims = len(tile)
    #print(f"samples dtype:{samples.dtype}")
    output = torch.empty(
        [samples.shape[0], out_channels] + list(map(lambda a: round(a * upscale_amount), samples.shape[2:])),
        device=output_device,
    )

    for b in range(samples.shape[0]):
        s = samples[b : b + 1]
        out = torch.zeros(
            [s.shape[0], out_channels] + list(map(lambda a: round(a * upscale_amount), s.shape[2:])),
            device=output_device,
        )
        out_div = torch.zeros(
            [s.shape[0], out_channels] + list(map(lambda a: round(a * upscale_amount), s.shape[2:])),
            device=output_device,
        )

        for it in itertools.product(*map(lambda a: range(0, a[0], a[1] - overlap), zip(s.shape[2:], tile))):
            s_in = s
            upscaled = []

            for d in range(dims):
                pos = max(0, min(s.shape[d + 2] - overlap, it[d]))
                l = min(tile[d], s.shape[d + 2] - pos)
                s_in = s_in.narrow(d + 2, pos, l)
                upscaled.append(round(pos * upscale_amount))

            ps = function(s_in).to(output_device)
            mask = torch.ones_like(ps)
            feather = round(overlap * upscale_amount)
            for t in range(feather):
                for d in range(2, dims + 2):
                    m = mask.narrow(d, t, 1)
                    m *= (1.0 / feather) * (t + 1)
                    m = mask.narrow(d, mask.shape[d] - 1 - t, 1)
                    m *= (1.0 / feather) * (t + 1)

            o = out
            o_d = out_div
            for d in range(dims):
                o = o.narrow(d + 2, upscaled[d], mask.shape[d + 2])
                o_d = o_d.narrow(d + 2, upscaled[d], mask.shape[d + 2])

            o += ps * mask
            o_d += mask

            if pbar is not None:
                pbar.update(1)

        output[b : b + 1] = out / out_div
    return output


def tiled_scale(
    samples,
    function,
    tile_x=64,
    tile_y=64,
    overlap=8,
    upscale_amount=4,
    out_channels=3,
    output_device="cpu",
    pbar=None,
):
    return tiled_scale_multidim(
        samples, function, (tile_y, tile_x), overlap, upscale_amount, out_channels, output_device, pbar
    )


def load_sd_upscale(ckpt, inf_device):
    sd = load_torch_file(ckpt, device=inf_device)
    if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
        sd = state_dict_prefix_replace(sd, {"module.": ""})
    out = ModelLoader().load_from_state_dict(sd).half()
    return out


def upscale(upscale_model, tensor: torch.Tensor, inf_device, output_device="cpu") -> torch.Tensor:
    """Upscale a tensor using the provided model.
    
    Args:
        upscale_model: The upscaling model to use
        tensor: Input tensor of shape [F,C,H,W] where F can be batch or frames
        inf_device: Device to run inference on
        output_device: Device for output tensor
        
    Returns:
        Upscaled tensor
    """
    logger.debug(f"Upscaling tensor of shape: {tensor.shape}")
    
    memory_required = module_size(upscale_model.model)
    memory_required += (
        (512 * 512 * 3) * tensor.element_size() * max(upscale_model.scale, 1.0) * 384.0
    )
    memory_required += tensor.nelement() * tensor.element_size()
    logger.debug(f"UPScaleMemory required: {memory_required / 1024 / 1024 / 1024} GB")

    upscale_model.to(inf_device)
    tile = 512
    overlap = 32

    # tensor.shape = [F,C,H,W] so we use indices 2 and 3 for height and width
    steps = tensor.shape[0] * get_tiled_scale_steps(
        tensor.shape[3],  # Width is last dimension
        tensor.shape[2],  # Height is second to last
        tile_x=tile,
        tile_y=tile,
        overlap=overlap
    )

    # we hide the progressbar otherwise it bloats the logs
    #pbar = ProgressBar(steps, desc="Tiling and Upscaling")
    pbar = None

    try:
        s = tiled_scale(
            samples=tensor.to(torch.float16),
            function=lambda a: upscale_model(a),
            tile_x=tile,
            tile_y=tile,
            overlap=overlap,
            upscale_amount=upscale_model.scale,
            pbar=pbar
        )
    finally:
        upscale_model.to(output_device)
    
    return s

def upscale_batch_and_concatenate(upscale_model, latents, inf_device, output_device="cpu") -> torch.Tensor:
    """Process frames individually for upscaling to manage memory usage.
    
    Args:
        upscale_model: The upscaling model to use
        latents: Input tensor of shape [F,C,H,W] where F is number of frames
        inf_device: Device to run inference on
        output_device: Device for output tensor
        
    Returns:
        Upscaled tensor with shape [F,C,H*scale,W*scale]
    """
    logger.debug(f"Processing latents shape: {latents.shape}")
    
    upscaled_latents = []
    for i in range(latents.size(0)):
        logger.debug(f"Processing frame {i+1}/{latents.size(0)}")
        # Process one frame at a time for memory efficiency
        latent = latents[i:i+1]  # Keep dimensions as [1,C,H,W]
        upscaled_latent = upscale(upscale_model, latent, inf_device, output_device)
        upscaled_latents.append(upscaled_latent)
    
    result = torch.cat(upscaled_latents, dim=0)
    logger.debug(f"Final upscaled shape: {result.shape}")
    return result

class ProgressBar:
    def __init__(self, total, desc=None):
        self.total = total
        self.current = 0
        self.b_unit = tqdm.tqdm(total=total, desc="ProgressBar context index: 0" if desc is None else desc)

    def update(self, value):
        if value > self.total:
            value = self.total
        self.current = value
        if self.b_unit is not None:
            self.b_unit.set_description("ProgressBar context index: {}".format(self.current))
            self.b_unit.refresh()

            # 更新进度
            self.b_unit.update(self.current)
