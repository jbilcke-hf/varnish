from varnish import Varnish, MMAudioConfig

async def process_video():
    # Initialize Varnish
    varnish = Varnish(
        enable_mmaudio=True,
        mmaudio_config=MMAudioConfig(
        prompt="nature sounds",
        negative_prompt="music",
        seed=42
    )
    )

    # Process video
    def progress_callback(progress):
        print(f"{progress.stage}: {progress.progress * 100}% - {progress.message}")

    result = await varnish(
        "input.mp4",
        input_fps=24,
        output_fps=60,
        enable_upscale=True,
        enable_interpolation=True,
        target_width=1920,
        target_height=1080,
        grain_amount=5.0,
        progress_callback=progress_callback
    )

    # Save result
    await result.write(
        output_type="file",
        output_filename="enhanced.mp4"
    )

# Run the async function
asyncio.run(process_video())