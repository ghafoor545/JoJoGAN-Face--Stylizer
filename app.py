import gradio as gr
from PIL import Image
import torch
import numpy as np
from util import align_face
from e4e_projection import projection as e4e_projection
from torchvision import transforms
from model import Generator
import os
import subprocess

latent_dim = 512
device = 'cuda' if torch.cuda.is_available() else 'cpu'
weights_dir = "pretrained_weights"

# Pretrained model IDs from JoJoGAN
drive_ids = {
    "art.pt": "1a0QDEHwXQ6hE_FcYEyNMuv5r5UnRQLKT",
    "arcane_multi.pt": "15V9s09sgaw-zhKp116VHigf5FowAy43f",
    "sketch_multi.pt": "1GdaeHGBGjBAFsWipTL0y-ssUiAqk8AxD",
    "jojo.pt": "13cR2xjIBj8Ga5jMO7gtxzIJj2PDsBYK4",
    "disney.pt": "1zbE2upakFUAx8ximYnLofFwfT8MilqJA"
}

transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

def download_ckpt(style_name):
    os.makedirs(weights_dir, exist_ok=True)
    ckpt_path = os.path.join(weights_dir, f"{style_name}.pt")
    if not os.path.exists(ckpt_path):
        file_id = drive_ids.get(f"{style_name}.pt")
        if file_id is None:
            raise ValueError("Invalid style selected.")
        subprocess.run(["gdown", "--id", file_id, "-O", ckpt_path], check=True)
    return ckpt_path

def stylize_face(input_image_path, style_name):
    ckpt_path = download_ckpt(style_name)
    generator = Generator(1024, latent_dim, 8, 2).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    generator.load_state_dict(ckpt["g"], strict=False)
    generator.eval()

    aligned = align_face(input_image_path)
    my_w = e4e_projection(aligned, "temp_input.pt", device).unsqueeze(0)

    with torch.no_grad():
        output = generator(my_w, input_is_latent=True)

    out = output[0].detach().cpu()
    out = 1 + out
    out /= 2
    out = 255 * torch.clip(out, 0, 1)
    out = out.permute(1, 2, 0).byte().numpy()

    return Image.fromarray(out)

style_choices = list(drive_ids.keys())
style_labels = [name.replace(".pt", "") for name in style_choices]

interface = gr.Interface(
    fn=stylize_face,
    inputs=[
        gr.Image(type="filepath", label="Upload Face Image"),
        gr.Dropdown(choices=style_labels, label="Select Style")
    ],
    outputs=gr.Image(label="Stylized Output"),
    title="JoJoGAN Face Stylizer",
    description="Upload a face and select a style. This app downloads the appropriate pretrained JoJoGAN model and stylizes your face."
)

interface.launch(share=True)
