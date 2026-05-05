"""
Tokenize the downloaded WAV clips with EnCodec for the nano4M-Audio extension.

Input:  {raw_dir}/{split}/{class}/{ytid}_{start}.wav     (24 kHz, mono PCM16, 1.71 s)
Output: {out_dir}/{split}/tok_audio@256/{ytid}_{start}.npy   shape (1, 256), int16, ids in [0, 1023]
        {out_dir}/{split}/scene_desc/{ytid}_{start}.json     ["<class>"]   (JSON list, K augmentations)

EnCodec config:
  - 24 kHz model @ 1.5 kbps  ->  K=2 RVQ codebooks at 75 Hz
  - 40,960 input samples (= 128 * 320)  ->  exactly 128 timesteps with no waste
  - K=2, T=128  ->  256 tokens per clip (matches the nano4M-Audio token budget)
  - Flatten order: [C1[0], C2[0], C1[1], C2[1], ..., C1[127], C2[127]]
    (interleaved RVQ; keeps both codebooks at each timestep adjacent)
  - Tokens saved as raw EnCodec ids in [0, 1023]; the unified-vocab offset is
    applied later by nano4M's `to_unified_multimodal_vocab` at dataset load
    time (overlap_vocab=False).

nano4M compatibility (matches `SimpleMultimodalDataset`):
  - The `@256` suffix is part of the directory name; the dataloader reads
    `{root}/{split}/{modality}/{stem}{ext}` where the modality string in the
    config is literally "tok_audio@256", not "tok_audio".
  - scene_desc/*.json is a JSON LIST of strings (one per augmentation), not a
    dict. The dataloader does captions[augmentation_idx]; we use a single-
    element list ["<class>"] and run with sample_from_k_augmentations=1.

Why exactly 40,960 samples (and not int(1.71 * 24000) = 41,040)?
  EnCodec produces one frame per 320 samples, so the trailing 80 samples of a
  41,040-sample clip silently round down to the same 128 frames anyway. Trim/
  pad to 40,960 first to make the (1, 256) shape exact and avoid 80 samples
  (3.3 ms) of audio being dropped at encode time.

Usage (run from this repo's root, after `python download_person2.py ...`):
  python tokenize_audio_encodec.py --raw_dir data/raw --out_dir data/tokenized
  python tokenize_audio_encodec.py --raw_dir data/raw --out_dir data/tokenized --device cuda
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SAMPLE_RATE       = 24_000
FRAME_RATE        = 75
SAMPLES_PER_FRAME = SAMPLE_RATE // FRAME_RATE        # 320
N_CODEBOOKS       = 2
N_TIMESTEPS       = 128
TARGET_SAMPLES    = N_TIMESTEPS * SAMPLES_PER_FRAME  # 40_960 -> exactly 128 frames
N_TOKENS          = N_CODEBOOKS * N_TIMESTEPS        # 256
BANDWIDTH         = 1.5                              # kbps -> 2 codebooks


def load_model(device: str = "cpu"):
    from encodec import EncodecModel
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(BANDWIDTH)
    model.to(device)
    model.eval()
    return model


def tokenize_wav(wav_path: Path, model, device: str = "cpu") -> np.ndarray:
    data, sr = sf.read(str(wav_path), dtype="float32")
    wav = torch.from_numpy(data)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)                                # [1, T]
    else:
        wav = wav.T.mean(dim=0, keepdim=True)                 # stereo -> mono [1, T]

    if sr != SAMPLE_RATE:
        wav = AF.resample(wav, sr, SAMPLE_RATE)

    if wav.shape[1] < TARGET_SAMPLES:
        wav = torch.nn.functional.pad(wav, (0, TARGET_SAMPLES - wav.shape[1]))
    elif wav.shape[1] > TARGET_SAMPLES:
        wav = wav[:, :TARGET_SAMPLES]

    wav = wav.unsqueeze(0).to(device)                         # [1, 1, T]
    with torch.no_grad():
        encoded_frames = model.encode(wav)
    codes = encoded_frames[0][0].squeeze(0)                   # [K, T] = [2, 128]

    codes = codes[:N_CODEBOOKS, :N_TIMESTEPS]                 # defensive no-op for clean inputs

    tokens = codes.cpu().numpy().T.flatten().astype(np.int16) # [256], values in [0, 1023]
    return tokens[np.newaxis, :]                              # [1, 256]


def class_to_caption(cls_dir_name: str) -> str:
    return cls_dir_name.replace("_", " ").strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/raw",       help="Root of downloaded WAV/JPG data")
    parser.add_argument("--out_dir", default="data/tokenized", help="Root for tokenized output")
    parser.add_argument("--splits",  nargs="+", default=["train", "test"])
    parser.add_argument("--device",  default="cpu", choices=["cpu", "mps", "cuda"],
                        help="Device for EnCodec encode")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    log.info(f"Loading EnCodec 24kHz model on device={args.device}...")
    model = load_model(device=args.device)
    log.info(f"EnCodec ready (bw={BANDWIDTH} kbps, K={N_CODEBOOKS}, "
             f"{N_TIMESTEPS} frames @ {FRAME_RATE} Hz, {TARGET_SAMPLES} input samples/clip)")

    wav_files = []
    for split in args.splits:
        split_dir = raw_dir / split
        if not split_dir.exists():
            log.warning(f"Split dir not found: {split_dir}")
            continue
        for cls_dir in sorted(split_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            for wav_path in sorted(cls_dir.glob("*.wav")):
                wav_files.append((split, cls_dir.name, wav_path))

    log.info(f"Found {len(wav_files)} WAV files across splits {args.splits}")

    ok = skip = fail = 0
    start_time = time.time()

    with tqdm(total=len(wav_files), unit="file",
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
        for split, cls_dir_name, wav_path in wav_files:
            s = wav_path.stem

            tok_dir  = out_dir / split / "tok_audio@256"
            desc_dir = out_dir / split / "scene_desc"
            tok_dir.mkdir(parents=True, exist_ok=True)
            desc_dir.mkdir(parents=True, exist_ok=True)

            npy_out  = tok_dir  / f"{s}.npy"
            json_out = desc_dir / f"{s}.json"

            if npy_out.exists() and json_out.exists():
                skip += 1
                pbar.set_postfix(ok=ok, skip=skip, fail=fail)
                pbar.update(1)
                continue

            try:
                tokens = tokenize_wav(wav_path, model, device=args.device)
                if tokens.shape != (1, N_TOKENS) or tokens.dtype != np.int16:
                    raise ValueError(f"unexpected token shape/dtype: {tokens.shape} {tokens.dtype}")
                if int(tokens.min()) < 0 or int(tokens.max()) > 1023:
                    raise ValueError(f"token id out of range [0, 1023]: "
                                     f"min={int(tokens.min())} max={int(tokens.max())}")
                np.save(str(npy_out), tokens)

                caption = class_to_caption(cls_dir_name)
                with open(json_out, "w") as f:
                    json.dump([caption], f)

                ok += 1
            except Exception as e:
                fail += 1
                log.warning(f"FAIL {s}: {e}")

            pbar.set_postfix(ok=ok, skip=skip, fail=fail)
            pbar.update(1)

    elapsed = time.time() - start_time
    log.info(f"Done -- OK:{ok}  SKIP:{skip}  FAIL:{fail}  Time:{elapsed/60:.1f} min")
    log.info(f"Output: {out_dir}/")
    log.info(f"  tok_audio@256/  -> .npy shape [1, {N_TOKENS}], dtype int16, ids in [0, 1023]")
    log.info(f"  scene_desc/     -> .json list of K captions  e.g. [\"<class name>\"]")


if __name__ == "__main__":
    main()
