from __future__ import annotations

from contextlib import nullcontext
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

import av
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

from .utils import load_sd_upscale, upscale_batch_and_concatenate
from .rife_model import load_rife_model
from .debug_utils import verify_model_paths, log_directory_structure

from pathlib import Path

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        enable_mmaudio: bool = True,
        mmaudio_config: Optional[MMAudioConfig] = None,
        model_base_dir: Optional[str] = None
    ):
        self.device = device
        self._models: dict[str, Any] = {}
        
        # Get base directory for models
        if model_base_dir is None:
            # Try to find the varnish directory relative to the current file
            current_file = Path(__file__).resolve()
            model_base_dir = current_file.parent.parent / 'varnish'
            if not model_base_dir.exists():
                # Fall back to current working directory
                model_base_dir = Path.cwd() / 'varnish'
        
        # Log directory structure for debugging
        logger.debug("Logging directory structure for debugging...")
        log_directory_structure(str(Path(model_base_dir).parent))
        
        # Define relative model paths
        model_paths = {
            'upscale_x2': "real_esrgan/RealESRGAN_x2.pth",
            'upscale_x4': "real_esrgan/RealESRGAN_x4.pth",
            'upscale_x8': "real_esrgan/RealESRGAN_x8.pth",
            'mmaudio': "mmaudio",
            'rife': "rife/rife-flownet-4.13.2.safetensors"
        }
        
        # Verify and convert to absolute paths
        try:
            self.model_paths = verify_model_paths(model_paths, base_dir=model_base_dir)
        except Exception as e:
            logger.error(f"Error verifying model paths: {e}")
            raise RuntimeError(f"Failed to initialize VideoProcessor: {str(e)}")
        
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
            if model_type.startswith('upscale'):
                self._models[model_type] = load_sd_upscale(
                    self.model_paths[model_type], 
                    self.device
                )
            elif model_type == 'rife':
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
        upscale_factor: Optional[float] = None,
        enable_interpolation: bool = False,
        target_fps: Optional[int] = None,
        progress_callback: Optional[callable] = None
    ) -> tuple[torch.Tensor, VideoMetadata]:
        """Process video frames with optional upscaling and interpolation
        
        Args:
            frames: Input video frames tensor
            upscale_factor: Optional factor to upscale the video
            enable_interpolation: Whether to enable frame interpolation
            target_fps: Target output frame rate
            progress_callback: Optional callback for progress updates
            
        Returns:
            Tuple of (processed frames tensor, metadata)
        """
        try:
            # Reshape frames if needed - ensure BCHW format
            if len(frames.shape) == 3:  # CHW format
                frames = frames.unsqueeze(0)  # Add batch dimension
            elif len(frames.shape) == 5:  # NBCHW format
                frames = frames.squeeze(0)  # Remove batch dimension

            # Ensure frames are in correct format [batch, channels, height, width]
            if len(frames.shape) != 4:
                raise ValueError(f"Expected tensor of shape [frames, channels, height, width], got shape {frames.shape}")

            # Ensure frames are on the correct device and in the right format
            frames = frames.to(device=self.device, dtype=torch.float32)
            frames = frames / 255.0 if frames.max() > 1.0 else frames

            # Process upscaling if requested
            if upscale_factor and upscale_factor > 1:
                if progress_callback:
                    progress_callback(ProcessingProgress(
                        ProcessingStage.UPSCALING,
                        0.0,
                        "Starting upscaling"
                    ))
                    
                with torch.cuda.stream(self.upscale_stream) if self.device == "cuda" else nullcontext():
                    # Select appropriate upscale model based on factor
                    if upscale_factor <= 2:
                        model_key = 'upscale_x2'
                    elif upscale_factor <= 4:
                        model_key = 'upscale_x4'
                    else:
                        model_key = 'upscale_x8'
                        
                    model = self._load_model(model_key)
                    frames = await asyncio.to_thread(
                        upscale_batch_and_concatenate,
                        model,
                        frames,
                        self.device
                    )
                    
                    if progress_callback:
                        progress_callback(ProcessingProgress(
                            ProcessingStage.UPSCALING,
                            1.0,
                            "Upscaling complete"
                        ))

            # Process frame interpolation if requested
            if enable_interpolation:
                if progress_callback:
                    progress_callback(ProcessingProgress(
                        ProcessingStage.INTERPOLATION,
                        0.0,
                        "Starting frame interpolation"
                    ))
                    
                with torch.cuda.stream(self.rife_stream) if self.device == "cuda" else nullcontext():
                    rife_model = self._load_model('rife')
                    frames = await asyncio.to_thread(
                        rife_inference_with_latents,
                        rife_model,
                        frames,
                        target_fps
                    )
                    
                    if progress_callback:
                        progress_callback(ProcessingProgress(
                            ProcessingStage.INTERPOLATION,
                            1.0,
                            "Frame interpolation complete"
                        ))

            # Create metadata
            metadata = VideoMetadata(
                width=frames.shape[3],
                height=frames.shape[2],
                fps=target_fps or frames.shape[0],  # Use original fps if no target
                duration=frames.shape[0] / (target_fps or frames.shape[0]),
                frame_count=frames.shape[0]
            )
            
            return frames, metadata

        except Exception as e:
            logger.error(f"Error processing frames: {str(e)}")
            raise RuntimeError(f"Failed to process frames: {str(e)}")
    
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
        """Write processed video to specified format using PyAV"""
        if output_type == "file" and not output_filename:
            raise ValueError("output_filename is required for file output type")

        # Convert frames to numpy for PyAV
        frames_np = (self.frames.cpu().numpy() * 255).astype(np.uint8)
        frames_np = frames_np.transpose(0, 2, 3, 1)  # Convert from BCHW to BHWC

        # Create temporary file if needed
        if not self._temp_file:
            with tempfile.NamedTemporaryFile(suffix=f".{output_format}", delete=False) as tmp:
                self._temp_file = tmp.name

        # Open output container
        output = av.open(self._temp_file, mode='w')
        
        try:
            # Add video stream
            stream = output.add_stream(output_codec, rate=self.metadata.fps)
            stream.width = self.metadata.width
            stream.height = self.metadata.height
            stream.pix_fmt = 'yuv420p'
            
            # Set quality/bitrate
            if output_bitrate:
                # Convert string bitrate (e.g., "5M") to bits per second
                multiplier = {'k': 1000, 'K': 1000, 'm': 1000000, 'M': 1000000}
                number = float(re.match(r'(\d+)', output_bitrate).group(1))
                unit = output_bitrate[-1] if output_bitrate[-1] in multiplier else ''
                bitrate = int(number * multiplier.get(unit, 1))
                stream.bit_rate = bitrate
            else:
                # Use quality-based encoding
                stream.options = {'crf': str(output_quality)}

            # Add audio stream if available
            audio_stream = None
            if self.audio_path:
                audio_container = av.open(self.audio_path)
                audio_stream = output.add_stream(template=audio_container.streams.audio[0])

            # Write video frames
            for frame_idx, frame_data in enumerate(frames_np):
                frame = av.VideoFrame.from_ndarray(frame_data, format='rgb24')
                packet = stream.encode(frame)
                output.mux(packet)

            # Flush video stream
            packet = stream.encode(None)
            output.mux(packet)

            # Copy audio if available
            if audio_stream and self.audio_path:
                for packet in audio_container.demux():
                    if packet.dts is not None:
                        output.mux(packet)
                audio_container.close()

        finally:
            output.close()

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
        model_base_dir: Optional[str] = None,
    ):
        self.processor = VideoProcessor(
            device=device,
            enable_mmaudio=enable_mmaudio,
            mmaudio_config=mmaudio_config,
            model_base_dir=model_base_dir,
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
        upscale_factor: Optional[float] = None,
        enable_interpolation: bool = False,
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
            upscale_factor: Factor to upscale the video (e.g., 2.0 doubles dimensions)
            enable_interpolation: Whether to enable frame interpolation
            grain_amount: Amount of film grain to add (0-100)
            mmaudio_prompt: Text prompt for audio generation
            mmaudio_negative_prompt: Negative prompt for audio generation
            progress_callback: Optional callback for progress updates
        
        Returns:
            VarnishResult object containing processed video
        """
        try:
            if progress_callback:
                progress_callback(ProcessingProgress(
                    ProcessingStage.LOADING,
                    0.0,
                    "Loading input data"
                ))

            # Process video and generate audio in parallel
            async with asyncio.TaskGroup() as tg:
                # Video processing task
                video_task = tg.create_task(
                    self.processor.process_frames(
                        frames=input_data,
                        upscale_factor=upscale_factor,
                        enable_interpolation=enable_interpolation,
                        target_fps=output_fps,
                        progress_callback=progress_callback
                    )
                )
                
                # Audio generation task if enabled
                audio_task = None
                if self.processor.enable_mmaudio:
                    audio_task = tg.create_task(
                        self.processor.generate_audio(
                            input_data,
                            output_duration_in_sec or (len(input_data) / (output_fps or input_fps)),
                            progress_callback=progress_callback
                        )
                    )

            # Get results from parallel processing
            processed_frames, frames_metadata = await video_task
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
                    fps=output_fps or input_fps,
                    duration=output_duration_in_sec or (len(processed_frames) / (output_fps or input_fps)),
                    frame_count=len(processed_frames)
                ),
                audio_path=audio_path
            )

        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            raise RuntimeError(f"Failed to process video: {str(e)}")

    async def _load_video(
        self,
        video_input: Union[str, List, torch.Tensor],
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
        if isinstance(video_input, torch.Tensor):
            # Handle tensor input directly - assumes BCHW format
            if len(video_input.shape) == 4:  # [batch, channels, height, width]
                metadata = VideoMetadata(
                    width=video_input.shape[3],
                    height=video_input.shape[2],
                    fps=input_fps,
                    duration=video_input.shape[0] / input_fps,
                    frame_count=video_input.shape[0]
                )
                return video_input, metadata
            else:
                raise ValueError(f"Expected tensor of shape [frames, channels, height, width], got shape {video_input.shape}")
                
        elif isinstance(video_input, str):
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