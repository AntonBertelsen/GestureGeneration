
libs = {
    "ipynb": "from IPython import get_ipython",
    "numpy": "import numpy as np",
    "torch": "import torch",
    "torchaudio": "import torchaudio",
    "librosa": "import librosa",
    "matplotlib": "import matplotlib",
    "transformers": "import transformers",
    "datasets": "import datasets",
    "einops": "import einops",
    "local-attention": "import local_attention",
    "moviepy": "import moviepy.editor as mpy",
    "wandb": "import wandb",
}

for name, code in libs.items():
    try:
        exec(code)
        print(f"✅ {name} is installed.")
    except ImportError as e:
        print(f"❌ {name} is NOT installed: {e}")
    except Exception as e:
        print(f"⚠️ {name} raised an unexpected error: {e}")
