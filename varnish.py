from __future__ import annotations

import asyncio
import base64
import io
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, List, Literal, Optional, Union, AsyncGenerator

import cv2
import ffmpeg
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
import torchaudio
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip

import mmaudio
from mmaudio.eval_utils import (
    ModelConfig,
    all_model_cfg,
    generate,
    load_video,
    make_video,
    setup_eval_logging
)
from mmaudio.model.flow_matching import FlowMatching
from mmaudio.model.networks import MMAudio, get_my_mmaudio
from mmaudio.model.sequence_config import SequenceConfig
from mmaudio.model.utils.features_utils import FeaturesUtils

# Type definitions
PipelineImageInput = Union[
    str,  # File path or base64 data URI
    PIL.Image.Image,
    np.ndarray,
    torch.Tensor,
    List[PIL.Image.Image],
    List[np.ndarray],
    List[torch.Tensor],
    List[str],  # List of file paths or base64 URIs
]

OutputType = Literal["file", "data-uri", "binary"]

class ProcessingStage(Enum):
    """Enum for different processing stages to track progress"""
    LOADING = "loading"
    UPSCALING = "upscaling"
    INTERPOLATION = "interpolation"
    AUDIO_GENERATION = "audio_generation"
    ENCODING = "encoding"

@dataclass
class VideoMetadata:
    """Metadata for video processing"""
    width: int
    height: int
    fps: float
    duration: float
    frame_count: int

@dataclass
class MMAudioConfig:
    """Configuration for MMAudio processing"""
    prompt: str = ""
    negative_prompt: str = "music"
    seed: int = 0
    num_steps: int = 25
    cfg_strength: float = 4.5
    model_name: str = 'large_44k_v2'

@dataclass
class ProcessingProgress:
    """Track processing progress"""
    stage: ProcessingStage
    progress: float  # 0.0 to 1.0
    message: str

class VideoProcessor:
    """Core video processing class"""
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        upscale_model_path: str = "model_real_esran/RealESRGAN_x4.pth",
        rife_model_path: str = "model_rife",
        enable_mmaudio: bool = True,
        mmaudio_config: Optional[MMAudioConfig] = None,
    ):
        self.device = device
        self._models: dict[str, Any] = {}
        self.model_paths = {
            'upscale': upscale_model_path,
            'rife': rife_model_path
        }
        self.enable_mmaudio = enable_mmaudio
        self.mmaudio_config = mmaudio_config or MMAudioConfig()
        
        # Initialize CUDA streams for parallel processing
        if device == "cuda":
            self.upscale_stream = torch.cuda.Stream()
            self.rife_stream = torch.cuda.Stream()
            self.mmaudio_stream = torch.cuda.Stream()
        
        if self.enable_mmaudio:
            self._setup_mmaudio()

    def _setup_mmaudio(self) -> None:
        """Initialize MMAudio models and utilities"""
        with torch.cuda.stream(self.mmaudio_stream) if self.device == "cuda" else nullcontext():
            model = all_model_cfg[self.mmaudio_config.model_name]
            model.download_if_needed()
            
            seq_cfg = model.seq_cfg
            net: MMAudio = get_my_mmaudio(model.model_name).to(self.device, torch.bfloat16).eval()
            net.load_weights(torch.load(model.model_path, map_location=self.device, weights_only=True))
            
            feature_utils = FeaturesUtils(
                tod_vae_ckpt=model.vae_path,
                synchformer_ckpt=model.synchformer_ckpt,
                enable_conditions=True,
                mode=model.mode,
                bigvgan_vocoder_ckpt=model.bigvgan_16k_path,
                need_vae_encoder=False
            ).to(self.device, torch.bfloat16).eval()
            
            self._models['mmaudio'] = {
                'net': net,
                'utils': feature_utils,
                'seq_cfg': seq_cfg
            }

    def _load_model(self, model_type: str) -> Any:
        """Lazy load models when needed"""
        if model_type not in self._models:
            if model_type == 'upscale':
                from utils import load_sd_upscale
                self._models[model_type] = load_sd_upscale(
                    self.model_paths[model_type], 
                    self.device
                )
            elif model_type == 'rife':
                from rife_model import load_rife_model
                self._models[model_type] = load_rife_model(
                    self.model_paths[model_type]
                )
        return self._models[model_type]

    async def _resize_for_mmaudio(self, frames: torch.Tensor) -> torch.Tensor:
        """Resize frames to have max 384px on shortest side for MMAudio"""
        _, _, height, width = frames.shape
        
        if min(height, width) <= 384:
            return frames
            
        if width < height:
            new_width = 384
            new_height = int((height * 384) / width)
        else:
            new_height = 384
            new_width = int((width * 384) / height)
            
        return F.interpolate(
            frames,
            size=(new_height, new_width),
            mode='bilinear',
            align_corners=False
        )

    async def _generate_audio(
        self,
        frames: torch.Tensor,
        duration: float,
        progress_callback: Optional[callable] = None
    ) -> str:
        """Generate audio using MMAudio"""
        if progress_callback:
            progress_callback(ProcessingProgress(
                ProcessingStage.AUDIO_GENERATION,
                0.0,
                "Preparing audio generation"
            ))

        with torch.cuda.stream(self.mmaudio_stream) if self.device == "cuda" else nullcontext():
            # Resize frames for MMAudio
            resized_frames = await self._resize_for_mmaudio(frames)
            
            # Setup MMAudio generation
            rng = torch.Generator(device=self.device)
            rng.manual_seed(self.mmaudio_config.seed)
            
            fm = FlowMatching(
                min_sigma=0,
                inference_mode='euler',
                num_steps=self.mmaudio_config.num_steps
            )
            
            if progress_callback:
                progress_callback(ProcessingProgress(
                    ProcessingStage.AUDIO_GENERATION,
                    0.3,
                    "Generating audio"
                ))
            
            # Generate audio
            net = self._models['mmaudio']['net']
            utils = self._models['mmaudio']['utils']
            seq_cfg = self._models['mmaudio']['seq_cfg']
            
            seq_cfg.duration = duration
            net.update_seq_lengths(
                seq_cfg.latent_seq_len,
                seq_cfg.clip_seq_len,
                seq_cfg.sync_seq_len
            )
            
            audios = generate(
                resized_frames,
                None,  # sync frames
                [self.mmaudio_config.prompt],
                negative_text=[self.mmaudio_config.negative_prompt],
                feature_utils=utils,
                net=net,
                fm=fm,
                rng=rng,
                cfg_strength=self.mmaudio_config.cfg_strength
            )
            
            if progress_callback:
                progress_callback(ProcessingProgress(
                    ProcessingStage.AUDIO_GENERATION,
                    0.8,
                    "Saving audio"
                ))
            
            # Save audio to temporary file
            audio = audios.float().cpu()[0]
            with tempfile.NamedTemporaryFile(suffix='.flac', delete=False) as tmp:
                torchaudio.save(tmp.name, audio, seq_cfg.sampling_rate)
                if progress_callback:
                    progress_callback(ProcessingProgress(
                        ProcessingStage.AUDIO_GENERATION,
                        1.0,
                        "Audio generation complete"
                    ))
                return tmp.name

    async def process_frames(
        self,
        frames: torch.Tensor,
        enable_upscale: bool = False,
        enable_interpolation: bool = False,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
        target_fps: Optional[int] = None,
        progress_callback: Optional[callable] = None
    ) -> torch.Tensor:
        """Process video frames with optional upscaling and interpolation"""
        processed_frames = frames

        if enable_upscale and (target_width is not None or target_height is not None):
            if progress_callback:
                progress_callback(ProcessingProgress(
                    ProcessingStage.UPSCALING,
                    0.0,
                    "Starting upscaling"
                ))
                
            with torch.cuda.stream(self.upscale_stream) if self.device == "cuda" else nullcontext():
                model = self._load_model('upscale')
                processed_frames = utils.upscale_batch_and_concatenate(
                    model,
                    processed_frames,
                    self.device
                )
                
                if target_width is not None or target_height is not None:
                    processed_frames = F.interpolate(
                        processed_frames,
                        size=(
                            target_height or processed_frames.shape[2],
                            target_width or processed_frames.shape[3]
                        ),
                        mode='bicubic',
                        align_corners=False
                    )
                
                if progress_callback:
                    progress_callback(ProcessingProgress(
                        ProcessingStage.UPSCALING,
                        1.0,
                        "Upscaling complete"
                    ))

        if enable_interpolation and target_fps is not None:
            if progress_callback:
                progress_callback(ProcessingProgress(
                    ProcessingStage.INTERPOLATION,
                    0.0,
                    "Starting frame interpolation"
                ))
                
            with torch.cuda.stream(self.rife_stream) if self.device == "cuda" else nullcontext():
                model = self._load_model('rife')
                processed_frames = rife_model.rife_inference_with_latents(
                    model,
                    processed_frames
                )
                
                if progress_callback:
                    progress_callback(ProcessingProgress(
                        ProcessingStage.INTERPOLATION,
                        1.0,
                        "Frame interpolation complete"
                    ))

        return processed_frames

class VarnishResult:
    """Handle processed video results and output generation"""
    def __init__(
        self,
        frames: torch.Tensor,
        metadata: VideoMetadata,
        audio_path: Optional[str] = None,
        temp_file: Optional[str] = None
    ):
        self.frames = frames
        self.metadata = metadata
        self.audio_path = audio_path
        self._temp_file = temp_file

    async def write(
        self,
        output_type: OutputType,
        output_filename: Optional[str] = None,
        output_format: str = "mp4",
        output_codec: str = "h264",
        output_quality: int = 23,
        output_bitrate: Optional[str] = None,
    ) -> Union[str, bytes, bool]:
        """Write processed video to specified format"""
        if output_type == "file" and not output_filename:
            raise ValueError("output_filename is required for file output type")

        # Prepare video data
        frames_np = self.frames.cpu().numpy()
        
        # Create temporary file if needed
        if not self._temp_file:
            with tempfile.NamedTemporaryFile(suffix=f".{output_format}", delete=False) as tmp:
                self._temp_file = tmp.name

        # Prepare FFmpeg command
        stream = ffmpeg.input(
            'pipe:',
            format='rawvideo',
            pix_fmt='rgb24',
            s=f'{self.metadata.width}x{self.metadata.height}',
            r=self.metadata.fps
        )
        
        # Add audio if available
        if self.audio_path:
            audio_stream = ffmpeg.input(self.audio_path)
            stream = ffmpeg.concat(stream, audio_stream, v=1, a=1)
        
        stream = ffmpeg.output(
            stream,
            self._temp_file,
            vcodec=output_codec,
            crf=output_quality,
            **({'b:v': output_bitrate} if output_bitrate else {})
        )
        
        await asyncio.to_thread(ffmpeg.run, stream, capture_stdout=True, capture_stderr=True)

        if output_type == "file":
            os.rename(self._temp_file, output_filename)
            return True
        elif output_type == "data-uri":
            with open(self._temp_file, "rb") as f:
                video_bytes = f.read()
            return f"data:video/mp4;base64,{base64.b64encode(video_bytes).decode()}"
        else:  # binary
            with open(self._temp_file, "rb") as f:
                return f.read()

    def __del__(self):
        """Cleanup temporary files"""
        if self._temp_file and os.path.exists(self._temp_file):
            os.unlink(self._temp_file)
        if self.audio_path and os.path.exists(self.audio_path):
            os.unlink(self.audio_path)

class Varnish:
    """Main interface for video processing"""
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        output_format: str = "mp4",
        output_codec: str = "h264",
        output_quality: int = 23,
        enable_mmaudio: bool = True,
        mmaudio_config: Optional[MMAudioConfig] = None,
    ):
        self.processor = VideoProcessor(
            device=device,
            enable_mmaudio=enable_mmaudio,
            mmaudio_config=mmaudio_config
        )
        self.default_output_format = output_format
        self.default_output_codec = output_codec
        self.default_output_quality = output_quality

    async def __call__(
        self,
        input_data: PipelineImageInput,
        input_fps: Optional[int] = 24,
        output_duration_in_sec: Optional[float] = None,
        output_fps: Optional[int] = None,
        enable_upscale: bool = False,
        enable_interpolation: bool = False,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
        grain_amount: float = 0.0,
        mmaudio_prompt: Optional[str] = None,
        mmaudio_negative_prompt: Optional[str] = None,
        progress_callback: Optional[callable] = None,
    ) -> VarnishResult:
        """
        Process video with optional enhancements and audio generation.
        
        Args:
            input_data: Input video or image sequence
            input_fps: Input frame rate
            output_duration_in_sec: Desired output duration
            output_fps: Desired output frame rate
            enable_upscale: Whether to enable upscaling
            enable_interpolation: Whether to enable frame interpolation
            target_width: Desired output width
            target_height: Desired output height
            grain_amount: Amount of film grain to add (0-100)
            mmaudio_prompt: Text prompt for audio generation
            mmaudio_negative_prompt: Negative prompt for audio generation
            progress_callback: Optional callback for progress updates
        
        Returns:
            VarnishResult object containing processed video
        """
        if progress_callback:
            progress_callback(ProcessingProgress(
                ProcessingStage.LOADING,
                0.0,
                "Loading input data"
            ))

        # Load input data
        frames, metadata = await self._load_video(input_data, input_fps)
        
        # Update metadata with input parameters
        if input_fps:
            metadata.fps = input_fps
            metadata.duration = metadata.frame_count / input_fps

        # Calculate final duration after potential frame interpolation
        final_fps = output_fps or metadata.fps
        final_frame_count = frames.shape[0]
        if enable_interpolation and output_fps:
            final_frame_count *= (output_fps / metadata.fps)
        final_duration = final_frame_count / final_fps

        # Update MMAudio config if prompts provided
        if mmaudio_prompt is not None:
            self.processor.mmaudio_config.prompt = mmaudio_prompt
        if mmaudio_negative_prompt is not None:
            self.processor.mmaudio_config.negative_prompt = mmaudio_negative_prompt

        # Process video and generate audio in parallel
        async with asyncio.TaskGroup() as tg:
            # Video processing task
            video_task = tg.create_task(
                self.processor.process_frames(
                    frames,
                    enable_upscale=enable_upscale,
                    enable_interpolation=enable_interpolation,
                    target_width=target_width,
                    target_height=target_height,
                    target_fps=output_fps,
                    progress_callback=progress_callback
                )
            )
            
            # Audio generation task if enabled
            audio_task = None
            if self.processor.enable_mmaudio:
                audio_task = tg.create_task(
                    self.processor.generate_audio(
                        frames,
                        final_duration,
                        progress_callback=progress_callback
                    )
                )

        # Get results from parallel processing
        processed_frames = await video_task
        audio_path = await audio_task if audio_task else None

        # Apply film grain if requested
        if grain_amount > 0:
            noise = torch.randn_like(processed_frames) * (grain_amount / 100.0)
            processed_frames = torch.clamp(processed_frames + noise, 0, 1)

        if progress_callback:
            progress_callback(ProcessingProgress(
                ProcessingStage.ENCODING,
                1.0,
                "Processing complete"
            ))

        return VarnishResult(
            frames=processed_frames,
            metadata=VideoMetadata(
                width=processed_frames.shape[3],
                height=processed_frames.shape[2],
                fps=output_fps or metadata.fps,
                duration=output_duration_in_sec or metadata.duration,
                frame_count=processed_frames.shape[0]
            ),
            audio_path=audio_path
        )

    async def _load_video(
        self,
        video_input: Union[str, List],
        input_fps: int
    ) -> tuple[torch.Tensor, VideoMetadata]:
        """
        Load video data from various input formats.
        
        Args:
            video_input: Input video data or path
            input_fps: Input frame rate
            
        Returns:
            Tuple of (frames tensor, video metadata)
        """
        if isinstance(video_input, str):
            if self._is_base64(video_input):
                # Handle base64 video
                return await self._load_base64_video(video_input, input_fps)
            else:
                # Handle file path
                return await self._load_file_video(video_input, input_fps)
        elif isinstance(video_input, (list, tuple)):
            # Handle frame sequence
            return await self._load_frame_sequence(video_input, input_fps)
        else:
            raise ValueError(f"Unsupported video input type: {type(video_input)}")

    @staticmethod
    def _is_base64(s: str) -> bool:
        """Check if string is base64 encoded"""
        return bool(re.match(r'^data:[^;]+;base64,', s))

    async def _load_base64_video(
        self,
        base64_data: str,
        input_fps: int
    ) -> tuple[torch.Tensor, VideoMetadata]:
        """Load video from base64 string"""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            data = re.sub(r'^data:[^;]+;base64,', '', base64_data)
            tmp.write(base64.b64decode(data))
            tmp_path = tmp.name
        
        try:
            return await self._load_file_video(tmp_path, input_fps)
        finally:
            os.unlink(tmp_path)

    async def _load_file_video(
        self,
        file_path: str,
        input_fps: int
    ) -> tuple[torch.Tensor, VideoMetadata]:
        """Load video from file path"""
        clip = VideoFileClip(file_path)
        
        frames = []
        async for frame in self._iter_video_frames(clip):
            frames.append(torch.from_numpy(frame).permute(2, 0, 1))
        
        metadata = VideoMetadata(
            width=clip.size[0],
            height=clip.size[1],
            fps=clip.fps,
            duration=clip.duration,
            frame_count=len(frames)
        )
        
        clip.close()
        return torch.stack(frames), metadata

    async def _iter_video_frames(self, clip: VideoFileClip) -> AsyncGenerator[np.ndarray, None]:
        """Asynchronously iterate over video frames"""
        for frame in clip.iter_frames():
            yield frame

    async def _load_frame_sequence(
        self,
        frames: List[Union[str, PIL.Image.Image, np.ndarray, torch.Tensor]],
        input_fps: int
    ) -> tuple[torch.Tensor, VideoMetadata]:
        """Load video from sequence of frames"""
        processed_frames = []
        for frame in frames:
            if isinstance(frame, str):
                if self._is_base64(frame):
                    frame = await self._load_base64_image(frame)
                else:
                    frame = PIL.Image.open(frame)
            processed_frames.append(self._process_frame(frame))
        
        frames_tensor = torch.stack(processed_frames)
        metadata = VideoMetadata(
            width=frames_tensor.shape[3],
            height=frames_tensor.shape[2],
            fps=input_fps,
            duration=len(frames) / input_fps,
            frame_count=len(frames)
        )
        
        return frames_tensor, metadata

    async def _load_base64_image(self, base64_data: str) -> PIL.Image.Image:
        """Load image from base64 string"""
        data = re.sub(r'^data:[^;]+;base64,', '', base64_data)
        image_bytes = base64.b64decode(data)
        return PIL.Image.open(io.BytesIO(image_bytes))

    def _process_frame(
        self,
        frame: Union[PIL.Image.Image, np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """Convert frame to tensor format"""
        if isinstance(frame, PIL.Image.Image):
            return torch.from_numpy(np.array(frame)).permute(2, 0, 1)
        elif isinstance(frame, np.ndarray):
            return torch.from_numpy(frame).permute(2, 0, 1)
        elif isinstance(frame, torch.Tensor):
            return frame
        else:
            raise ValueError(f"Unsupported frame type: {type(frame)}")