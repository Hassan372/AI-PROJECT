"""
Tokenize RGB JPGs with Cosmos-0.1-Tokenizer-DI16x16 for the nano4M-Audio
extension.

Input:  {raw_dir}/{split}/{class}/{ytid}_{start}.jpg
Output: {out_dir}/{split}/tok_rgb@256/{ytid}_{start}.npy   shape (1, 256), int32

Preprocessing matches nano4M's notebook (COM304_FM_part3_nano4M.ipynb) EXACTLY:
    img = Image.open(p).resize((256, 256)).convert('RGB')
    img_tensor = TF.to_tensor(img).unsqueeze(0) * 2 - 1      # range [-1, 1]
This is critical: any deviation (different resize, missing centering, BGR/RGB
swap, etc.) puts the codes off-distribution for the Cosmos decoder and the
downstream nano4M-Audio model.

Cosmos DI16x16:  H=256 -> h=16, W=256 -> w=16  =>  tokens (B, 16, 16) int32, ids in [0, 63999]
We flatten to (1, 256) per image and save as int32 .npy. K=1 augmentations
(no synthetic augmentations -- the nano4M config sets
sample_from_k_augmentations=1).

Performance notes (~1,700 JPGs):
  - CUDA L40S/A100, bfloat16, batch 32-64:  ~5-10 min   <- canonical run path
  - CUDA T4/V100,   bfloat16, batch 16:     ~10-20 min
  - Apple Silicon CPU, bfloat16, batch 1:   ~16 h (CPU bf16 is sw-emulated)
  - Apple Silicon MPS:                       not supported -- the Cosmos JIT
                                             traces float64 buffers MPS rejects;
                                             tested May 2026.

Usage (on a CUDA box, after `python download_person*.py ...`):
  python tokenize_rgb_cosmos.py --device cuda --batch_size 32

The Cosmos checkpoint (~340 MB) is auto-downloaded to
~/.cache/cosmos/Cosmos-0.1-Tokenizer-DI16x16/ on first run; pass
--checkpoint_dir to override.
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

GRID         = 16            # 16x16 token grid for DI16x16
N_TOKENS     = GRID * GRID   # 256
IMG_SIZE     = 256
COSMOS_VOCAB = 64_000


def load_image_tokenizer(checkpoint_dir: Path, device: str, dtype: str):
    from cosmos_tokenizer.image_lib import ImageTokenizer
    return ImageTokenizer(
        checkpoint_enc=str(checkpoint_dir / "encoder.jit"),
        checkpoint_dec=str(checkpoint_dir / "decoder.jit"),
        device=device,
        dtype=dtype,
    )


def preprocess(jpg_path: Path) -> torch.Tensor:
    """JPG -> float32 (3, 256, 256) tensor in [-1, 1], matching nano4M notebook."""
    img = Image.open(jpg_path).resize((IMG_SIZE, IMG_SIZE)).convert("RGB")
    return TF.to_tensor(img) * 2 - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="data/raw")
    ap.add_argument("--out_dir", default="data/tokenized")
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument(
        "--checkpoint_dir",
        default=str(Path.home() / ".cache/cosmos/Cosmos-0.1-Tokenizer-DI16x16"),
        help="Local dir with encoder.jit / decoder.jit (auto-downloaded if missing)",
    )
    ap.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="MPS not supported by Cosmos JIT (float64 buffers); tested May 2026",
    )
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    ckpt = Path(args.checkpoint_dir)

    if not (ckpt / "encoder.jit").exists():
        log.info(f"Cosmos checkpoint not found at {ckpt}; downloading from HuggingFace...")
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="nvidia/Cosmos-0.1-Tokenizer-DI16x16",
            local_dir=str(ckpt),
        )

    log.info(f"Loading Cosmos DI16x16  device={args.device}  dtype={args.dtype}")
    tok = load_image_tokenizer(ckpt, args.device, args.dtype)
    torch_dtype = getattr(torch, args.dtype)
    log.info("Loaded.")

    jobs = []
    for split in args.splits:
        split_dir = raw_dir / split
        if not split_dir.exists():
            log.warning(f"Split dir not found: {split_dir}")
            continue
        for cls_dir in sorted(split_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            for jpg in sorted(cls_dir.glob("*.jpg")):
                jobs.append((split, jpg))
    log.info(f"Found {len(jobs)} JPGs across splits {args.splits}")

    for split in args.splits:
        (out_dir / split / "tok_rgb@256").mkdir(parents=True, exist_ok=True)

    ok = skip = fail = 0
    start = time.time()

    pending: list[torch.Tensor] = []
    pending_meta: list[tuple[str, str, Path]] = []

    def flush():
        nonlocal ok, fail
        if not pending:
            return
        try:
            batch = torch.stack(pending).to(torch_dtype).to(args.device)
            with torch.no_grad():
                out = tok.encode(batch)
            tokens = out[0] if isinstance(out, tuple) else out
            tokens = tokens.to("cpu").numpy().astype(np.int32)
            assert tokens.shape == (len(pending), GRID, GRID), tokens.shape
            for i, (_split, stem, npy_path) in enumerate(pending_meta):
                arr = tokens[i].reshape(1, N_TOKENS)
                if int(arr.min()) < 0 or int(arr.max()) >= COSMOS_VOCAB:
                    log.warning(
                        f"FAIL {stem}: id out of [0, {COSMOS_VOCAB}): "
                        f"min={int(arr.min())} max={int(arr.max())}"
                    )
                    fail += 1
                    continue
                np.save(str(npy_path), arr)
                ok += 1
        except Exception as e:
            log.warning(f"FAIL batch ({len(pending)} imgs): {e}")
            fail += len(pending)
        pending.clear()
        pending_meta.clear()

    pbar = tqdm(
        total=len(jobs),
        unit="img",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )
    for split, jpg in jobs:
        stem = jpg.stem
        npy_path = out_dir / split / "tok_rgb@256" / f"{stem}.npy"
        if npy_path.exists():
            skip += 1
        else:
            try:
                pending.append(preprocess(jpg))
                pending_meta.append((split, stem, npy_path))
                if len(pending) >= args.batch_size:
                    flush()
            except Exception as e:
                log.warning(f"FAIL preprocess {stem}: {e}")
                fail += 1
        pbar.update(1)
        pbar.set_postfix(ok=ok, skip=skip, fail=fail)

    if pending:
        flush()
        pbar.set_postfix(ok=ok, skip=skip, fail=fail)
    pbar.close()

    elapsed = time.time() - start
    log.info(f"Done -- OK:{ok}  SKIP:{skip}  FAIL:{fail}  Time:{elapsed/60:.1f} min")
    log.info(
        f"Output: {out_dir}/{{split}}/tok_rgb@256/  shape (1, {N_TOKENS}) int32, "
        f"ids in [0, {COSMOS_VOCAB - 1}]"
    )


if __name__ == "__main__":
    main()
