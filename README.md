# Varnish 💅

A comprehensive video enhancement library with AI-powered audio generation.

*WARNING: this is a beta and experimental project and it may break at any time*

*AI WARNING NOTE: I don't have the time to personally develop Varnish - I need it for a ton of projects, but Varnish is not really a "project" in itself - so I used Claude 3.5 to generate all the code*

## Introduction

The goal of Varnish is NOT to generate videos using AI, but to improve existing videos that have been generated using AI.

It does things that (at the time of writing) many models don't do, such as adding a soundtrack, adding film grain, controlling the duration and FPS precisely.

## Acknowledgements

This project uses various open-source projects to work.

All credits is due to them! Varnish is only a modest library mixing and mashing things together.

In particular Varnish uses those AI projects:

- MMAudio: https://github.com/hkchengrex/MMAudio
- Real-ESRGAN: https://github.com/xinntao/Real-ESRGAN
- RIFE: https://github.com/hzwer/ECCV2022-RIFE

Please see the `requirements.txt` file for details.

## Installation

### From PyPI
```bash
# ..not yet!
# pip install varnish
```

### From Source
```bash
git clone https://github.com/jbilcke-hf/varnish
cd varnish
pip install -e .
```

## Basic Usage

```python
import asyncio
from varnish import Varnish

async def enhance_video():
    # Initialize Varnish
    varnish = Varnish(
        enable_mmaudio=True  # Enable AI audio generation
    )
    
    # Process video
    result = await varnish(
        "input.mp4",
        input_fps=24,
        output_fps=60,
        enable_upscale=True,
        enable_interpolation=True,
        target_width=1920,
        target_height=1080,
        mmaudio_prompt="nature sounds, birds chirping"
    )
    
    # Save result
    await result.write(
        output_type="file",
        output_filename="enhanced.mp4"
    )

# Run the async function
asyncio.run(enhance_video())
```

## Examples

### 1. Process Video File with Audio Generation

```python
from varnish import Varnish, MMAudioConfig

# Configure MMAudio
mmaudio_config = MMAudioConfig(
    prompt="waves crashing, seagulls",
    negative_prompt="music, talking",
    seed=42,
    cfg_strength=4.5
)

# Initialize with custom config
varnish = Varnish(
    enable_mmaudio=True,
    mmaudio_config=mmaudio_config
)

async def process_video_file():
    result = await varnish(
        "beach.mp4",
        output_fps=60,
        enable_upscale=True,
        target_width=3840,
        target_height=2160
    )
    await result.write(output_type="file", output_filename="beach_enhanced.mp4")
```

### 2. Process Base64 Data URI

```python
async def process_base64_video():
    # Your base64 video data
    video_data_uri = "data:video/mp4;base64,..."
    
    result = await varnish(
        video_data_uri,
        enable_interpolation=True,
        output_fps=60,
        grain_amount=5.0  # Add 5% film grain
    )
    
    # Get result as base64
    output_uri = await result.write(output_type="data-uri")
```

### 3. Process Image Sequence

```python
import PIL.Image
import numpy as np

async def process_image_sequence():
    # List of image files
    image_files = ["frame1.jpg", "frame2.jpg", "frame3.jpg"]
    
    # Or list of PIL images
    pil_images = [PIL.Image.open(f) for f in image_files]
    
    # Or numpy arrays
    numpy_frames = [np.array(img) for img in pil_images]
    
    result = await varnish(
        numpy_frames,
        input_fps=24,
        enable_upscale=True,
        mmaudio_prompt="dramatic orchestral music"
    )
    
    await result.write(output_type="file", output_filename="sequence.mp4")
```

### 4. Progress Tracking

```python
async def process_with_progress():
    def progress_callback(progress):
        print(f"{progress.stage}: {progress.progress * 100:.1f}% - {progress.message}")
    
    result = await varnish(
        "input.mp4",
        enable_upscale=True,
        enable_interpolation=True,
        progress_callback=progress_callback
    )
```

### 5. Custom Video Encoding

```python
async def process_with_encoding_options():
    result = await varnish("input.mp4")
    
    # High quality encoding
    await result.write(
        output_type="file",
        output_filename="high_quality.mp4",
        output_codec="h264",
        output_quality=17,  # Lower is better quality
        output_bitrate="8M"
    )
```

## Advanced Configuration

### MMAudio Configuration
```python
from varnish import MMAudioConfig

config = MMAudioConfig(
    prompt="ambient nature sounds",
    negative_prompt="music, voices",
    seed=42,
    num_steps=25,
    cfg_strength=4.5,
    model_name='large_44k_v2'
)
```

### Video Processing Options
```python
result = await varnish(
    input_data="input.mp4",
    input_fps=24,            # Input frame rate
    output_fps=60,           # Target frame rate
    enable_upscale=True,     # Enable ESRGAN upscaling
    enable_interpolation=True, # Enable RIFE frame interpolation
    target_width=3840,       # Target width (4K)
    target_height=2160,      # Target height (4K)
    grain_amount=3.0,        # Add 3% film grain
    mmaudio_prompt="custom audio prompt",
    mmaudio_negative_prompt="unwanted sounds"
)
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please check out our [Contributing Guide](CONTRIBUTING.md) for guidelines.