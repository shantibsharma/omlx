---
library_name: mlx
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
pipeline_tag: image-text-to-text
base_model: google/gemma-4-26b-a4b-it
tags:
- mlx
---

# mlx-community/gemma-4-26b-a4b-it-8bit

This model was converted to MLX format from [`google/gemma-4-26b-a4b-it`](https://huggingface.co/google/gemma-4-26b-a4b-it)
using mlx-vlm version **0.4.3**.
Refer to the [original model card](https://huggingface.co/google/gemma-4-26b-a4b-it) for more details on the model.

## Use with mlx

```bash
pip install -U mlx-vlm
```

```bash
python -m mlx_vlm.generate --model mlx-community/gemma-4-26b-a4b-it-8bit --max-tokens 100 --temperature 0.0 --prompt "Describe this image." --image <path_to_image>
```
