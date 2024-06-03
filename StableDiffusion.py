import os
import torch
from diffusers import DiffusionPipeline

model_dir = "./stable_diffusion_model"

# Check if the model is already downloaded
if not os.path.exists(model_dir):
    # Download and save the model
    print("Downloading model...")
    pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    pipeline.save_pretrained(model_dir)
else:
    # Load the saved model
    print("Loading saved model...")
    pipeline = DiffusionPipeline.from_pretrained(model_dir)

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

    # Generate image
    with torch.no_grad():
        image = pipeline(prompt).images[0]

    # Save the image
    output_file = f"output_{prompt.replace(' ', '_')[:50]}.png"
    image.save('./images/' + output_file)
    print(f"Image saved as {output_file}")



