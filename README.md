# Varnish 💅

## What is Varnish?

Varnish is a library to add some flair to sequence of frames generated with AI video models, to turn them into nice MP4 video files, with precise control over the compression levels, framerate, film grain noise.

It can handle frame upscaling and frame interpolation.

## Funding

VideoModelStudio is 100% open-source project, this is a project I really develop just for myself initially, for my Hugging Face Inference Endpoints.

I'm more or less the only user so I maintain it on my personal time, so if you like it let me know 🫶

<a href="https://www.buymeacoffee.com/flngr" target="_blank"><img src="https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png" alt="Buy Me A Coffee" style="height: 41px !important;width: 174px !important;box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;-webkit-box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;" ></a>

## Acknowledgements

This project uses various open-source librairies and AI models to work.

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
