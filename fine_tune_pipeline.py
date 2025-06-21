# TASK 01: JoJoGAN Fine-Tuning Pipeline (Colab-Compatible Python Version)

import os
import glob
import torch
from torchvision import transforms, utils
from util import align_face, strip_path_extension, display_image
from e4e_projection import projection as e4e_projection
from model import Generator, Discriminator
from PIL import Image
from tqdm import tqdm
import numpy as np
from torch import optim
from torch.nn import functional as F

# Configuration
latent_dim = 512
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Downloaded models assumed to be in 'models/'
os.makedirs('models', exist_ok=True)
os.makedirs('test_input', exist_ok=True)
os.makedirs('style_images', exist_ok=True)
os.makedirs('style_images_aligned', exist_ok=True)
os.makedirs('inversion_codes', exist_ok=True)
os.makedirs('stylized_results', exist_ok=True)

# Load pretrained StyleGAN2 Generator
original_generator = Generator(1024, latent_dim, 8, 2).to(device)
ckpt = torch.load('models/stylegan2-ffhq-config-f.pt', map_location=device)
original_generator.load_state_dict(ckpt['g_ema'], strict=False)
mean_latent = original_generator.mean_latent(10000)
generator = original_generator

# Image transform
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Invert input faces
input_images = glob.glob("test_input/*.jpg")
for input_path in input_images:
    filename = os.path.basename(input_path)
    name = strip_path_extension(filename)
    latent_path = f'inversion_codes/{name}.pt'

    aligned_face = align_face(input_path)
    my_w = e4e_projection(aligned_face, latent_path, device).unsqueeze(0)
    torch.save({'latent': my_w.squeeze(0)}, latent_path)

# Align and invert style images
style_names = ["Style 2.jpg"]
targets, latents = [], []
for style_name in style_names:
    style_path = f'style_images/{style_name}'
    name = strip_path_extension(style_name)

    style_aligned = align_face(style_path)
    style_aligned.save(f'style_images_aligned/{name}.png')
    latent = e4e_projection(style_aligned, f'inversion_codes/{name}.pt', device)
    targets.append(transform(style_aligned).to(device))
    latents.append(latent.to(device))

targets = torch.stack(targets, 0)
latents = torch.stack(latents, 0)

# Fine-tune generator
alpha = 1.0
preserve_color = True
num_iter = 500

discriminator = Discriminator(1024, 2).eval().to(device)
discriminator.load_state_dict(ckpt['d'], strict=False)

generator = Generator(1024, latent_dim, 8, 2).to(device)
generator.load_state_dict(ckpt["g_ema"], strict=False)
g_optim = optim.Adam(generator.parameters(), lr=2e-3, betas=(0.0, 0.99))
id_swap = [9, 11, 15, 16, 17] if preserve_color else list(range(7, generator.n_latent))

for idx in tqdm(range(num_iter)):
    mean_w = generator.get_latent(torch.randn([latents.size(0), latent_dim]).to(device)).unsqueeze(1).repeat(1, generator.n_latent, 1)
    in_latent = latents.clone()
    in_latent[:, id_swap] = (1-alpha)*mean_w[:, id_swap] + alpha*latents[:, id_swap]

    img = generator(in_latent, input_is_latent=True)
    real_feat = discriminator(targets)
    fake_feat = discriminator(img)

    loss = sum([F.l1_loss(a, b) for a, b in zip(fake_feat, real_feat)]) / len(fake_feat)
    g_optim.zero_grad()
    loss.backward()
    g_optim.step()

# Stylize input faces
all_rows = []
for input_path in input_images:
    filename = os.path.basename(input_path)
    name = strip_path_extension(filename)

    aligned_face = align_face(input_path)
    my_w = e4e_projection(aligned_face, f"inversion_codes/{name}.pt", device).unsqueeze(0)
    with torch.no_grad():
        generator.eval()
        my_sample = generator(my_w, input_is_latent=True)

    output_image = my_sample[0].detach().cpu()
    output_image = 1 + output_image
    output_image /= 2
    output_image = 255 * torch.clip(output_image, 0, 1)
    output_image = output_image.permute(1, 2, 0).byte().numpy()
    pil_output = Image.fromarray(output_image)
    pil_output.save(f"stylized_results/{name}_stylized.jpg")

    input_tensor = transform(aligned_face).unsqueeze(0).to(device)
    targets_with_batch = [target.unsqueeze(0) for target in targets]
    row_images = torch.cat([*targets_with_batch, input_tensor, my_sample], dim=0)
    all_rows.append(row_images)

final_grid = torch.cat(all_rows, dim=0)
final_image = utils.make_grid(final_grid, normalize=True, value_range=(-1, 1), nrow=targets.shape[0] + 2)
final_np = final_image.permute(1, 2, 0).mul(255).clamp(0, 255).byte().cpu().numpy()
Image.fromarray(final_np).save("final_results_grid.jpg")
print("✅ All stylized outputs saved as final_results_grid.jpg")
