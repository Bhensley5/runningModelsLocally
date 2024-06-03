import os
import torch
import imageio
import numpy as np
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

# Set model name and local directory
model_name = "damo-vilab/text-to-video-ms-1.7b"
model_dir = "./text_to_video_model"

# Check if the model is already downloaded
if not os.path.exists(model_dir):
    # Download and save the model
    print("Downloading model...")
    pipeline = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16, variant='fp16')
    pipeline.save_pretrained(model_dir)
    
else:
    # Load the saved model
    print("Loading saved model...")
    pipeline = DiffusionPipeline.from_pretrained(model_dir, torch_dtype=torch.float16, variant='fp16')

# Move the model to GPU if available
if torch.cuda.is_available():
    pipeline.to("cuda")

print("Model is ready. Type your prompt or 'STOP' to end.")

while True:
    # Get user input
    prompt = input("Enter a prompt: ")

    if prompt.strip().upper() == "STOP":
        print("Stopping the program.")
        break
    
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    
    # optimize for GPU memory
    pipeline.enable_model_cpu_offload()
    pipeline.enable_vae_slicing()

    video_frames = pipeline(prompt, num_inference_steps=25, num_frames=200).frames
    video_frames_np = [np.array(frame) for frame in video_frames]
    video_frames_np = np.concatenate(video_frames_np, axis=0)

   # Save the video frames as a video file
    output_file = f"output_{prompt.replace(' ', '_')[:50]}.mp4"
    imageio.mimsave('./videos/' + output_file, video_frames_np, fps=24)
