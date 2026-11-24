from pathlib import Path
import json
import os
import subprocess
import tempfile

import gradio as gr
import torch
from modules import scripts
from modules.processing import DecodedSamples
try:
    from ldm_patched.modules import latent_formats, model_management
except ModuleNotFoundError as error:
    if error.name != "ldm_patched":
        raise
    latent_formats = None


class Script(scripts.Script):
    def title(self):
        return "PiD decoder"

    def show(self, is_img2img):
        return scripts.AlwaysVisible if latent_formats is not None else None

    def ui(self, is_img2img):
        with gr.Accordion("PiD decoder", open=False):
            enabled = gr.Checkbox(label="Enable PiD (4x output)", value=False)
        self.infotext_fields = [
            (enabled, lambda params: bool(params.get("PiD decoder")))
        ]
        return [enabled]

    def post_sample(self, p, ps, enabled):
        if latent_formats is None or not enabled or getattr(ps.samples, "already_decoded", False):
            return
        model = p.sd_model
        if not getattr(model, "is_sdxl", False):
            return
        fmt = model.forge_objects.unet.model.model_config.latent_format
        if type(fmt) is not latent_formats.SDXL:
            return
        repo = Path(os.environ.get("NVIDIA_PID_ROOT", "")).expanduser().resolve()
        exe = Path(os.environ.get("NVIDIA_PID_PYTHON", "")).expanduser().resolve()
        if not (repo / "pid/_src/inference/decoder.py").is_file() or not exe.is_file():
            raise ValueError("Set NVIDIA_PID_ROOT and NVIDIA_PID_PYTHON before starting WebUI.")
        latent = ps.samples
        if latent is None or latent.ndim != 4 or latent.shape[1] != 4:
            raise ValueError("PiD expects normalized single-image BCHW latents.")
        count = latent.shape[0]
        start = p.iteration * p.batch_size
        prompts = p.all_prompts[start:start + count]
        seeds = p.all_seeds[start:start + count]
        if len(prompts) != count or len(seeds) != count:
            raise ValueError("Missing per-image prompt or seed.")
        with tempfile.TemporaryDirectory(prefix="reforge-pid-") as tmp:
            folder = Path(tmp)
            torch.save(latent.detach().cpu(), folder / "latent.pt")
            (folder / "batch.json").write_text(json.dumps(list(zip(prompts, seeds))), encoding="utf-8")
            model_management.unload_all_models()
            worker = Path(__file__).resolve().parents[1] / "decode_worker.py"
            env = os.environ.copy()
            env["PYTHONPATH"] = str(repo)
            subprocess.run([str(exe), str(worker), str(folder)], cwd=repo, env=env, check=True)
            images = DecodedSamples()
            shape = (3, latent.shape[-2] * 32, latent.shape[-1] * 32)
            for index in range(count):
                rgb = torch.load(folder / f"{index}.pt", map_location="cpu", weights_only=True)
                if tuple(rgb.shape) != shape or not rgb.isfinite().all():
                    raise ValueError("Invalid PiD output.")
                images.append(rgb)
        ps.samples = images
        p.extra_generation_params["PiD decoder"] = "sdxl, 4 steps, 4x"
