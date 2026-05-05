"""
Dataset viewer — scroll through images/audio with CLIP and CLAP scores.

Usage:
  python viewer.py --raw_dir data/raw --split train

Keys:
  ← / →   navigate
  P        replay audio
  Q        quit
"""

import argparse
import wave
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

try:
    import sounddevice as sd
    AUDIO_PLAY = True
except ImportError:
    AUDIO_PLAY = False

try:
    import clip
    import torch
    _clip_model, _clip_prep = clip.load("ViT-B/32", device="cpu")
    CLIP_AVAILABLE = True
except Exception:
    CLIP_AVAILABLE = False

try:
    from msclap import CLAP as MSCLAP
    _clap = MSCLAP(version="2023", use_cuda=False)
    CLAP_AVAILABLE = True
except Exception:
    CLAP_AVAILABLE = False


def clip_score(jpg: Path, cls: str) -> float:
    if not CLIP_AVAILABLE:
        return float("nan")
    img = _clip_prep(Image.open(jpg)).unsqueeze(0)
    text = clip.tokenize([f"a photo of a {cls}"])
    with torch.no_grad():
        i = _clip_model.encode_image(img)
        t = _clip_model.encode_text(text)
        i /= i.norm(dim=-1, keepdim=True)
        t /= t.norm(dim=-1, keepdim=True)
    return float((i @ t.T).item())


def clap_score(wav: Path, cls: str) -> float:
    if not CLAP_AVAILABLE:
        return float("nan")
    try:
        ae = _clap.get_audio_embeddings([str(wav)])
        te = _clap.get_text_embeddings([cls])
        s  = _clap.compute_similarity(ae, te)
        return float(s.detach().cpu().numpy()[0, 0])
    except Exception:
        return float("nan")


def load_wav(wav: Path):
    with wave.open(str(wav)) as w:
        sr  = w.getframerate()
        raw = w.readframes(w.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="data/raw")
    ap.add_argument("--split",   default="train")
    args = ap.parse_args()

    root = Path(args.raw_dir) / args.split
    items = []
    for cls_dir in sorted(root.iterdir()):
        if not cls_dir.is_dir():
            continue
        cls = cls_dir.name.replace("_", " ")
        for jpg in sorted(cls_dir.glob("*.jpg")):
            wav = jpg.with_suffix(".wav")
            if wav.exists():
                items.append((jpg, wav, cls))

    if not items:
        print("No data found.")
        return

    print(f"Found {len(items)} samples. ← → to navigate, P to replay, Q to quit.")

    idx   = [0]
    cache = {}

    fig = plt.figure(figsize=(13, 6))
    gs  = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[3, 1])
    ax_img  = fig.add_subplot(gs[0, 0])
    ax_wav  = fig.add_subplot(gs[0, 1])
    ax_info = fig.add_subplot(gs[1, :])
    ax_info.axis("off")

    def show(i):
        jpg, wav, cls = items[i]

        if i not in cache:
            print(f"  Computing scores for {jpg.name}...")
            cache[i] = (clip_score(jpg, cls), clap_score(wav, cls))
        cs, as_ = cache[i]

        ax_img.clear()
        ax_img.imshow(Image.open(jpg))
        ax_img.set_title(jpg.name, fontsize=9)
        ax_img.axis("off")

        ax_wav.clear()
        samples, sr = load_wav(wav)
        t = np.linspace(0, len(samples) / sr, len(samples))
        ax_wav.plot(t, samples, lw=0.5, color="steelblue")
        ax_wav.set_xlabel("Time (s)")
        ax_wav.set_title("Audio waveform", fontsize=9)
        ax_wav.set_ylim(-1, 1)

        ax_info.clear()
        ax_info.axis("off")
        ax_info.text(0.15, 0.5, f"Class:  {cls}", fontsize=12, va="center",
                     transform=ax_info.transAxes)
        ax_info.text(0.45, 0.5,
                     f"CLIP: {cs:.3f}" if not np.isnan(cs) else "CLIP: N/A",
                     fontsize=12, va="center", color="green" if cs >= 0.20 else "red",
                     transform=ax_info.transAxes)
        ax_info.text(0.70, 0.5,
                     f"CLAP: {as_:.3f}" if not np.isnan(as_) else "CLAP: N/A",
                     fontsize=12, va="center", color="green" if as_ >= 0.20 else "red",
                     transform=ax_info.transAxes)

        fig.suptitle(f"[{i+1}/{len(items)}]   ← →  navigate    P  replay    Q  quit",
                     fontsize=10)
        fig.canvas.draw_idle()

        if AUDIO_PLAY:
            sd.stop()
            sd.play(samples, sr)

    def on_key(event):
        if event.key == "right":
            idx[0] = (idx[0] + 1) % len(items)
            show(idx[0])
        elif event.key == "left":
            idx[0] = (idx[0] - 1) % len(items)
            show(idx[0])
        elif event.key == "p" and AUDIO_PLAY:
            samples, sr = load_wav(items[idx[0]][1])
            sd.stop()
            sd.play(samples, sr)
        elif event.key == "q":
            plt.close()

    fig.canvas.mpl_connect("key_press_event", on_key)
    show(0)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
