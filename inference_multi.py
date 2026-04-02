"""Multi-subject personalization with ARIA (LSI + GIA + CPD)."""

import os
import sys
import gc
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from pipeline import RFInversionParallelFluxPipeline
from utils import create_mask
from aria_processor import setup_aria_multi
from attn_processor import set_flux_transformer_attn_processor
from diffusers.models.attention_processor import FluxAttnProcessor2_0


MODEL_PATH = "black-forest-labs/FLUX.1-dev"  # or local path
IMG_SIZE = 1024
NUM_STEPS = 28
TAU = 0.90
GUIDANCE_SCALE = 3.5
ETA = 1.0


def load_pipe():
    pipe = RFInversionParallelFluxPipeline.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16)
    pipe.enable_sequential_cpu_offload()
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    _orig = type(pipe)
    class _P(_orig):
        @property
        def device(self): return torch.device("cuda")
        @property
        def _execution_device(self): return torch.device("cuda")
    pipe.__class__ = _P
    return pipe


def invert_subject(pipe, subject_dir, seed):
    img = Image.open(os.path.join(subject_dir, "background.png")).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    latent_h = IMG_SIZE // (pipe.vae_scale_factor * 2)
    latent_w = IMG_SIZE // (pipe.vae_scale_factor * 2)
    mask = create_mask(os.path.join(subject_dir, "mask.png"), w=latent_w, h=latent_h)
    img_dims = latent_h * latent_w

    def reset_proc(name, hidden_size, cross_attention_dim, ori_attn_proc):
        return FluxAttnProcessor2_0()
    set_flux_transformer_attn_processor(pipe.transformer, reset_proc)

    gen = torch.Generator(device="cuda").manual_seed(seed)
    inv, img_lat, lat_ids = pipe.invert(
        image=img, num_inversion_steps=NUM_STEPS, gamma=1.0,
        height=IMG_SIZE, width=IMG_SIZE, generator=gen)

    result = (mask, img_dims, inv.cpu(), img_lat.cpu(), lat_ids.cpu())
    del inv, img_lat, lat_ids
    torch.cuda.empty_cache()
    gc.collect()
    return result


def main():
    pipe = load_pipe()
    seed = 42

    # --- Invert two subjects ---
    subjects = ["example_data/white_cat", "example_data/teddy_bear"]
    data = {}
    for subj_dir in subjects:
        name = os.path.basename(subj_dir)
        print(f"Inverting {name}...")
        data[name] = invert_subject(pipe, subj_dir, seed)

    names = [os.path.basename(s) for s in subjects]
    masks = [data[n][0] for n in names]
    img_dims = data[names[0]][1]
    inverted = torch.cat([data[n][2] for n in names], dim=0).cuda()
    img_latents = torch.cat([data[n][3] for n in names], dim=0).cuda()
    lat_ids = data[names[0]][4].cuda()

    # --- Generate with ARIA (LSI + GIA) ---
    setup_aria_multi(pipe, [m.clone() for m in masks], img_dims, tau=TAU)

    prompt = "a white cat and a teddy bear playing on a grassy hill"
    gen = torch.Generator(device="cuda").manual_seed(seed)
    prompts = [""] * len(names) + [prompt]
    result = pipe(
        prompt=prompts,
        inverted_latents=inverted, image_latents=img_latents, latent_image_ids=lat_ids,
        start_timestep=0.0, stop_timestep=0.99,
        num_inference_steps=NUM_STEPS, eta=ETA, guidance_scale=GUIDANCE_SCALE,
        height=IMG_SIZE, width=IMG_SIZE, generator=gen)

    os.makedirs("outputs", exist_ok=True)
    result.images[-1].save("outputs/multi_subject_result.png")
    print("Saved to outputs/multi_subject_result.png")


if __name__ == "__main__":
    main()
