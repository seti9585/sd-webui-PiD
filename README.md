# sd-webui-PiD

An experimental final-latent decoder for reForge SDXL (including Illustrious). The Forge Neo adapter uses the Qwen-Image decoder for Anima and Qwen-Image (16 channels), and the FLUX.2 decoder for MOSS-Image (128 channels). Qwen-Image 2.1 uses a different VAE and is unsupported. Enable only one PiD extension per generation. PiD runs after sampling, independently of CFG extensions. Output width and height are each multiplied by four. Hardware decoding and extension combinations are unverified.

## Installation and use

1. Copy the sd-webui-PiD folder into the WebUI extensions folder.
2. Prepare a separate NVIDIA PiD checkout and CUDA Python environment using the official installation instructions. Run verify_env.py in that environment. Native Windows compatibility is unverified.
3. From the PiD checkout, download the weights for the intended model using the commands below. The official runtime may also need its text encoder assets.
4. Set NVIDIA_PID_ROOT to the checkout folder and NVIDIA_PID_PYTHON to the absolute path of its Python executable in the environment that starts WebUI.
5. Restart WebUI, open PiD decoder and enable PiD. Inspect the console on errors. The host can continue with ordinary decoding if the hook fails. Successful decoding adds PiD decoder to the image metadata.

SDXL:

```text
hf download nvidia/PiD --local-dir . --include "checkpoints/PiD_res2kto4k_sr4x_official_sdxl_distill_4step/*" "checkpoints/sdxl_vae.safetensors"
```

Anima / Qwen-Image (Forge Neo):

```text
hf download nvidia/PiD --local-dir . --include "checkpoints/PiD_v1pt5_res2kto4k_sr4x_official_qwenimage_distill_4step/*" "checkpoints/QwenImage_VAE_2d.pth"
```

MOSS-Image (Forge Neo):

```text
hf download nvidia/PiD --local-dir . --include "checkpoints/PiD_v1pt5_res2kto4k_sr4x_official_flux2_distill_4step/*" "checkpoints/flux2_ae.safetensors"
```

## 日本語

reForgeではSDXL（Illustriousを含む）の最終潜在データを4ステップでデコードします。Forge NeoではAnimaとQwen-ImageにQwen-Image用、MOSS-ImageにFLUX.2用の重みを使用します。Qwen-Image 2.1は対象外です。使用時は内蔵のPiD Integratedをオフにしてください。CFG系拡張は任意です。PiDはバッチごとに別プロセスで起動し、画像ごとのプロンプトとseedを使用します。

拡張フォルダーをWebUIのextensionsに配置します。公式手順で用意したPiD専用環境で verify_env.py を実行し、上記コマンドで対象モデルの重みを取得してください。起動環境の NVIDIA_PID_ROOT にPiD公式リポジトリの絶対パス、NVIDIA_PID_PYTHON に専用環境のPython実行ファイルの絶対パスを設定します。WebUIを再起動し、PiD decoderを有効にします。1024×1024入力では4096×4096出力を要求します。成功時はPNG生成情報に「PiD decoder」が記録されます。

## License, acknowledgments and sources

Based on the inference interface of [NVIDIA PiD](https://github.com/nv-tlabs/PiD/tree/2c8814c2b91cc41a2be7809962c891e0d0ccff5f). Official weights and their terms are available from [NVIDIA](https://huggingface.co/nvidia/PiD). The Anima VAE specification comes from [CircleStone Labs](https://huggingface.co/circlestone-labs/Anima). Upstream code and weights are distributed separately.

Reference: [ComfyUI PiD conditioning](https://github.com/Comfy-Org/ComfyUI/pull/14103).
