# Stable Diffusion 3.5 Image Generation Script

This script generates images using Stable Diffusion 3.5 based on prompts from a CSV file. It's designed to help evaluate cultural representation in AI image generation.

## Running Options

You can run this script either on your local machine or on a remote server. Choose the appropriate guide below based on your setup.

### Option 1: Local Machine Execution

If you have a GPU on your local machine, follow these steps:

1. **Set up Conda Environment**

    ```bash
    # Create and activate environment
    conda create -n sd35 python=3.10
    conda activate sd35

    # Install dependencies from requirements.txt
    pip install -r requirements.txt

    # Or install packages manually if needed:
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install diffusers transformers safetensors pandas
    ```

2. **Login to Hugging Face**

    ```bash
    pip install huggingface_hub
    huggingface-cli login
    ```

3. **Download Model**

    ```bash
    # Create directory and download
    mkdir -p ~/stable-diffusion-models/stable-diffusion-3.5-medium
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='stabilityai/stable-diffusion-3.5-medium', local_dir='~/stable-diffusion-models/stable-diffusion-3.5-medium')"
    ```

4. **Prepare Script and Prompts**

    - Modify paths in `general_culture_SD3.py`
    - Create `prompts.csv` with your prompts

5. **Generate Images**

    ```bash
    python general_culture_SD3.py
    ```

6. **View Results**
    - Check the `output` directory (or your specified output directory)
    - Images will be saved as PNG files with names based on the prompts

### Option 2: Server Execution

If you're running on a remote server, follow these steps:

1. **Connect to Server**

    ```bash
    ssh username@server-address
    # Example: ssh chans@big-rodn9.andrew.cmu.edu
    ```

2. **Set up Environment on Server**

    ```bash
    # Activate conda environment
    conda activate sd35
    ```

3. **Transfer Files to Server**

    ```bash
    # Upload script and prompts to server
    scp general_culture_SD3.py username@server-address:/path/to/destination/
    scp prompts.csv username@server-address:/path/to/destination/

    # Example:
    scp general_culture_SD3.py chans@big-rodn9.andrew.cmu.edu:/home/chans/worldccub/
    ```

4. **Run Script on Server**

    ```bash
    python general_culture_SD3.py
    ```

5. **Download Generated Images**

    ```bash
    # Option 1: Download entire output directory
    scp -r username@server-address:/path/to/output ./local_output

    # Option 2: Zip and download
    # On server:
    zip -r output.zip output
    # Then download:
    scp username@server-address:/path/to/output.zip ~/Downloads/

    # Example:
    scp chans@big-rodn9.andrew.cmu.edu:/scratch/chans/sd-data/output.zip ~/Downloads/
    ```

## Prompt File Format

Create a file named `prompts.csv` with your desired prompts:

```csv
prompt
"Show me traditional Nigerian clothing Fullbody"
"Show me traditional Nigerian celebration"
"Show me traditional Nigerian food"
"Show me traditional Nigerian architecture"
"Show me traditional Nigerian art"
```

## Command Line Arguments

-   `--model_path`: Path to Stable Diffusion model
-   `--prompts_csv`: Path to CSV file with prompts (default: "prompts.csv")
-   `--output_dir`: Directory to save generated images
-   `--steps`: Number of inference steps (default: 30)
-   `--guidance_scale`: Guidance scale for generation (default: 7.5)

## Troubleshooting

-   **CUDA errors**: If you have GPU memory issues, try reducing the `--steps` value
-   **Model loading failure**: Verify model path and Hugging Face login
-   **CSV file errors**: Check that your CSV file has "prompt" header
-   **Server connection issues**: Verify your SSH key and server address
-   **Permission denied**: Check file permissions (use `chmod +x` if needed)

## References

-   [Stable Diffusion 3.5 Medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium)
-   [Diffusers Documentation](https://huggingface.co/docs/diffusers/index)
-   [PyTorch Installation](https://pytorch.org/get-started/locally/)
