import torch
from diffusers import DiffusionPipeline
import os
import pandas as pd
import argparse

# ====== Configuration constants - modify before use ======

# Stable Diffusion 3.5 model path (location downloaded from Hugging Face)
SD_MODEL_PATH = "/scratch/chans/sd-models/stable-diffusion-3.5-medium"

# Output directory for generated images
DEFAULT_OUTPUT_DIR = "./output"

# Prompt CSV file path
DEFAULT_PROMPTS_CSV = "prompts.csv"


def generate_images(model_path, prompts_csv, output_dir, steps=30, guidance_scale=7.5):
    """
    Generate images using Stable Diffusion 3.5 based on prompts from a CSV file.

    Args:
        model_path: Path to the Stable Diffusion model
        prompts_csv: Path to CSV file containing prompts
        output_dir: Directory to save generated images
        steps: Number of inference steps
        guidance_scale: Guidance scale for generation
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print(f"Loading model from {model_path}...")
    pipe = DiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to("cuda")

    # Read prompts from CSV file
    print(f"Reading prompts from {prompts_csv}...")
    try:
        df = pd.read_csv(prompts_csv)
        if "prompt" not in df.columns:
            raise ValueError("CSV file must contain a 'prompt' column")
        prompts = df["prompt"].tolist()
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Generate images for each prompt
    print(f"Generating {len(prompts)} images...")
    for i, prompt in enumerate(prompts):
        print(f"\n🖼️ Generating image {i+1}/{len(prompts)}: {prompt}")

        # Generate image
        image = pipe(
            prompt, num_inference_steps=steps, guidance_scale=guidance_scale
        ).images[0]

        # Save image with index and sanitized prompt as filename
        # Remove special characters from prompt for filename
        safe_prompt = "".join(c if c.isalnum() or c in " _-" else "_" for c in prompt)
        safe_prompt = safe_prompt[:50]  # Limit filename length
        filename = f"{i+1:03d}_{safe_prompt}.png"

        # Save the image
        image.save(os.path.join(output_dir, filename))
        print(f"✅ Saved: {filename}")

    print("\n🎉 All images generated and saved!")


if __name__ == "__main__":
    # Set up command line arguments
    parser = argparse.ArgumentParser(
        description="Generate images using Stable Diffusion 3.5"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=SD_MODEL_PATH,
        help=f"Path to Stable Diffusion model (default: {SD_MODEL_PATH})",
    )
    parser.add_argument(
        "--prompts_csv",
        type=str,
        default=DEFAULT_PROMPTS_CSV,
        help=f"Path to CSV file containing prompts (default: {DEFAULT_PROMPTS_CSV})",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save generated images (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--steps", type=int, default=30, help="Number of inference steps"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Guidance scale for generation",
    )

    args = parser.parse_args()

    # Check if CSV exists
    if not os.path.exists(args.prompts_csv):
        print(f"Error: Prompts CSV file '{args.prompts_csv}' not found.")
        print("Please create a prompts.csv file with your desired prompts.")
        exit(1)

    # Generate images
    generate_images(
        args.model_path,
        args.prompts_csv,
        args.output_dir,
        args.steps,
        args.guidance_scale,
    )
