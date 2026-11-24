# sd-webui-PiD

A reForge extension that replaces the final SDXL VAE decode with NVIDIA PiD.
It uses the official 4-step SDXL decoder and produces images at four times
the width and height of a normal VAE decode. Illustrious uses the SDXL path.

PiD uses the final latent, the positive prompt as its caption, and sigma 0.
With Hires. fix enabled, PiD receives the result of the Hires pass. For example,
a final latent corresponding to a 1024 × 1024 image requests 4096 × 4096 output.
CFG and sampler extensions are optional.

## Installation and use

1. Place the sd-webui-PiD folder in reForge's extensions folder.
2. Follow the [NVIDIA PiD setup instructions](https://github.com/nv-tlabs/PiD#installation)
   to prepare a separate CUDA Python environment and run verify_env.py there.
3. From the PiD checkout, download the two files below. The upstream inference
   environment also needs its configured text encoder assets.
4. Before starting reForge, set NVIDIA_PID_ROOT to the PiD checkout and
   NVIDIA_PID_PYTHON to the full path of the Python executable in that environment.
5. Enable PiD decoder in the generation UI. Successful decoding records
   PiD decoder in the generation metadata. Check the console for errors.

```text
hf download nvidia/PiD --local-dir . --include "checkpoints/PiD_res2kto4k_sr4x_official_sdxl_distill_4step/model_ema_bf16.pth" "checkpoints/sdxl_vae.safetensors"
```

## 日本語

reForge専用のSDXL（Illustriousを含む）向け拡張です。画像生成の最後に、
通常のVAEデコードをNVIDIA PiDの4ステップデコードに置き換えます。
PiDには最終潜在データ、ポジティブプロンプト、sigma 0を渡します。
Hires. fixを有効にすると、その処理後の解像度を基準に縦横それぞれ4倍で出力します。
たとえばHires後が1024×1024相当なら、出力は4096×4096です。

拡張フォルダーをreForgeのextensionsに配置し、公式手順でPiD専用のPython環境を
用意してください。上記コマンドはPiD公式リポジトリのフォルダーで実行します。
起動環境の NVIDIA_PID_ROOT にPiD公式リポジトリの絶対パス、
NVIDIA_PID_PYTHON に専用環境のPython実行ファイルの絶対パスを設定します。
reForgeを起動し、PiD decoderを有効にすると使えます。
成功時は生成情報に PiD decoder が記録されます。

## License, acknowledgments and sources

The upstream [NVIDIA PiD code](https://github.com/nv-tlabs/PiD) is Apache-2.0.
NVIDIA distributes the [PiD model weights](https://huggingface.co/nvidia/PiD#licenseterms-of-use)
under NSCLv1. The weights and their derivatives are limited to non-commercial
research or evaluation. Download weights directly from NVIDIA. This extension
package contains no weights. Review the applicable terms before publishing
or using outputs commercially. The linked license does not explicitly address outputs.

公式PiDのコードはApache 2.0、モデル重みはNSCLv1です。重みとその派生物は
非商用の研究・評価用途に限られます。拡張には重みを同梱していないため、
NVIDIAから各自で取得してください。リンク先の規約には生成画像への適用を
明示した条項がないため、画像の公開や商用利用の前に適用条件を確認してください。
