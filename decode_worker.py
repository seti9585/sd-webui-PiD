import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from pid._src.inference.checkpoint_registry import get_pid_checkpoint
from pid._src.inference.decoder import load_our_decoder


@torch.inference_mode()
def main(folder, family):
    variant, vae = {
        "sdxl": ("2kto4k", "sdxl_vae.safetensors"),
        "flux2": ("2kto4k_v1pt5", "flux2_ae.safetensors"),
        "qwenimage": ("2kto4k_v1pt5", "QwenImage_VAE_2d.pth"),
    }[family]
    spec = get_pid_checkpoint(family, variant)
    for weight in (spec.checkpoint_path, Path("checkpoints") / vae):
        if not Path(weight).is_file():
            raise FileNotFoundError(f"Missing PiD weight: {weight}")
    options = SimpleNamespace(experiment=spec.experiment, checkpoint_path=spec.checkpoint_path,
                              config_file="pid/_src/configs/pid/config.py", load_ema_to_reg=True)
    decoder = load_our_decoder(options, [], True)
    latents = torch.load(folder / "latent.pt", map_location="cpu", weights_only=True)
    entries = json.loads((folder / "batch.json").read_text(encoding="utf-8"))
    for index, (prompt, seed) in enumerate(entries):
        latent = latents[index:index + 1].to(device="cuda", dtype=torch.bfloat16)
        batch = {decoder.config.input_caption_key: [prompt], "LQ_latent": latent,
                 "degrade_sigma": torch.zeros(1, device="cuda")}
        scale = 64 if family == "flux2" else 32
        result = decoder.generate_samples_from_batch(
            batch, cfg_scale=1.0, num_steps=4, seed=int(seed), shift=None,
            image_size=(latent.shape[-2] * scale, latent.shape[-1] * scale))
        if result.ndim == 5 and result.shape[2] == 1:
            result = result.squeeze(2)
        torch.save(result[0].float().cpu().clamp(-1, 1), folder / f"{index}.pt")


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2])

