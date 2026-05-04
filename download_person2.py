"""
VGGSound animal clips downloader — v3.

Pipeline:
  IMAGE : extract 20 frames evenly → batch CLIP → best frame → JPEG
  AUDIO : find 5 RMS peaks → CLAP score each 1.71s window → best → WAV

Output: <out_dir>/<split>/<class>/<ytid>_<start>.wav + .jpg

Usage:
  python download_vggsound.py --out_dir data/raw --workers 4
  python download_vggsound.py --out_dir data/raw --max_per_class 500
"""

import argparse
import csv
import logging
import shutil
import subprocess
import tempfile
import threading
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_model_lock = threading.Lock()

import numpy as np
from PIL import Image

try:
    import imageio_ffmpeg
    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    FFMPEG = "ffmpeg"

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

# ── Classes cibles ────────────────────────────────────────────────────────────
TARGET_CLASSES = {
    "cat meowing",
    "chimpanzee pant-hooting",
    "crow cawing",
    "donkey, ass braying",
    "owl hooting",
    "parrot talking",
}

CLIP_THRESHOLD  = 0.20
CLAP_THRESHOLD  = 0.20
N_FRAMES        = 20
N_PEAKS         = 5
AUDIO_DURATION  = 1.71

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def stem(ytid: str, start: int) -> str:
    return f"{ytid}_{start:06d}"


def extract_frames(tmp_mp4: Path, tmp: Path, n: int = 20) -> list[tuple[float, Path]]:
    """Extract N frames evenly spaced over 10s. Returns [(timestamp, path), ...]."""
    step = 10.0 / n
    result = []
    for i in range(n):
        t = step * i + step / 2
        fp = tmp / f"frame_{i:02d}.jpg"
        subprocess.run([
            FFMPEG, "-y", "-loglevel", "error",
            "-ss", str(t), "-i", str(tmp_mp4),
            "-frames:v", "1", "-q:v", "2", str(fp),
        ], capture_output=True)
        if fp.exists():
            result.append((t, fp))
    return result


def best_frame_clip(frames: list[tuple[float, Path]], cls: str) -> tuple[Path, float]:
    """Batch CLIP on all frames, return (best_path, best_score)."""
    if not frames:
        return None, 0.0
    if not CLIP_AVAILABLE:
        return frames[len(frames) // 2][1], 0.0

    images = torch.stack([_clip_prep(Image.open(fp)) for _, fp in frames])
    text = clip.tokenize([f"a photo of a {cls}"])
    with _model_lock, torch.no_grad():
        img_feats = _clip_model.encode_image(images)
        txt_feat  = _clip_model.encode_text(text)
        img_feats /= img_feats.norm(dim=-1, keepdim=True)
        txt_feat  /= txt_feat.norm(dim=-1, keepdim=True)
        scores = (img_feats @ txt_feat.T).squeeze()

    best_idx = int(scores.argmax())
    return frames[best_idx][1], float(scores[best_idx])


def find_top_peaks(wav_path: Path, n: int = 5, duration: float = 1.71, window_ms: int = 50) -> list[float]:
    """Return win_start offsets for the top N RMS peaks (min-separation enforced)."""
    with wave.open(str(wav_path)) as w:
        sr  = w.getframerate()
        raw = w.readframes(w.getnframes())
    frames = np.frombuffer(raw, dtype=np.int16).astype(np.float32)

    win      = max(1, int(sr * window_ms / 1000))
    rms      = np.array([np.sqrt(np.mean(frames[i:i+win]**2))
                         for i in range(0, len(frames)-win, win)])
    clip_dur = len(frames) / sr
    min_sep  = max(1, int(duration / (window_ms / 1000)))

    peaks, rms_work = [], rms.copy()
    for _ in range(n):
        if rms_work.max() == 0:
            break
        idx      = int(np.argmax(rms_work))
        peak_sec = idx * window_ms / 1000
        win_start = max(0.0, min(peak_sec - duration / 2, clip_dur - duration))
        peaks.append(win_start)
        rms_work[max(0, idx-min_sep): min(len(rms_work), idx+min_sep)] = 0

    return peaks


def best_audio_clap(tmp_mp4: Path, peaks: list[float], cls: str, tmp: Path) -> tuple[Path, float]:
    """Extract 1.71s WAV for each peak, CLAP score, return (best_path, best_score)."""
    wav_paths = []
    for i, win_start in enumerate(peaks):
        wp = tmp / f"audio_{i}.wav"
        subprocess.run([
            FFMPEG, "-y", "-loglevel", "error",
            "-ss", str(win_start), "-i", str(tmp_mp4),
            "-t", str(AUDIO_DURATION),
            "-vn", "-ar", "24000", "-ac", "1", "-sample_fmt", "s16", str(wp),
        ], capture_output=True)
        if wp.exists():
            wav_paths.append(wp)

    if not wav_paths:
        return None, 0.0

    if not CLAP_AVAILABLE:
        return wav_paths[0], 0.0

    try:
        with _model_lock:
            scores = _clap.compute_similarity([str(p) for p in wav_paths], [cls])
        best_idx = int(np.argmax(scores[:, 0]))
        return wav_paths[best_idx], float(scores[best_idx, 0])
    except Exception:
        return wav_paths[0], 0.0


def download_clip(ytid: str, start: int, cls: str, split: str, out_dir: Path) -> str:
    s          = stem(ytid, start)
    target_dir = out_dir / split / cls.replace("/", "_").replace(",", "")
    target_dir.mkdir(parents=True, exist_ok=True)

    wav_out = target_dir / f"{s}.wav"
    jpg_out = target_dir / f"{s}.jpg"

    if wav_out.exists() and jpg_out.exists():
        return f"SKIP {s}"

    url = f"https://www.youtube.com/watch?v={ytid}"

    with tempfile.TemporaryDirectory() as tmp:
        tmp       = Path(tmp)
        raw_video = tmp / "raw.%(ext)s"
        full_wav  = tmp / "full.wav"

        # ── 1. Download ───────────────────────────────────────────────────────
        r = subprocess.run([
            "yt-dlp", "--quiet", "--no-warnings",
            "--format", "best[ext=mp4][acodec!=none]/best[acodec!=none]",
            "--output", str(raw_video), "--no-playlist", url,
        ], capture_output=True, text=True)
        if r.returncode != 0:
            return f"FAIL_DL {s}: {r.stderr.strip()[:120]}"

        raw_files = list(tmp.glob("raw.*"))
        if not raw_files:
            return f"FAIL_DL {s}: no file"
        raw_file = raw_files[0]

        # ── 2. Cut 10s ────────────────────────────────────────────────────────
        tmp_mp4 = tmp / f"{s}.mp4"
        r = subprocess.run([
            FFMPEG, "-y", "-loglevel", "error",
            "-ss", str(start), "-i", str(raw_file), "-t", "10",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-c:a", "aac",
            str(tmp_mp4),
        ], capture_output=True, text=True)
        if r.returncode != 0:
            return f"FAIL_CUT {s}: {r.stderr.strip()[:120]}"

        # ── 3. Full 10s WAV ───────────────────────────────────────────────────
        r = subprocess.run([
            FFMPEG, "-y", "-loglevel", "error",
            "-i", str(tmp_mp4), "-vn", "-ar", "24000", "-ac", "1", "-sample_fmt", "s16",
            str(full_wav),
        ], capture_output=True, text=True)
        if r.returncode != 0:
            return f"FAIL_WAV_FULL {s}: {r.stderr.strip()[:120]}"

        # ── 4. IMAGE: 20 frames → batch CLIP → best ───────────────────────────
        frames = extract_frames(tmp_mp4, tmp, n=N_FRAMES)
        if not frames:
            return f"FAIL_FRAMES {s}"

        best_jpg, clip_score = best_frame_clip(frames, cls)
        if CLIP_AVAILABLE and clip_score < CLIP_THRESHOLD:
            return f"FAIL_CLIP {s}: clip={clip_score:.3f}"

        shutil.copy(best_jpg, jpg_out)

        # ── 5. AUDIO: 5 RMS peaks → CLAP → best 1.71s ────────────────────────
        peaks = find_top_peaks(full_wav, n=N_PEAKS)
        if not peaks:
            jpg_out.unlink(missing_ok=True)
            return f"FAIL_PEAKS {s}"

        best_wav, clap_score = best_audio_clap(tmp_mp4, peaks, cls, tmp)
        if best_wav is None:
            jpg_out.unlink(missing_ok=True)
            return f"FAIL_AUDIO {s}"

        if CLAP_AVAILABLE and clap_score < CLAP_THRESHOLD:
            jpg_out.unlink(missing_ok=True)
            return f"FAIL_CLAP {s}: clap={clap_score:.3f}"

        shutil.copy(best_wav, wav_out)

    return f"OK {s} | clip={clip_score:.3f} clap={clap_score:.3f}"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",           default="VGGSound/data/vggsound.csv")
    p.add_argument("--out_dir",       default="data/raw")
    p.add_argument("--split",         default="train", choices=["train", "test", "all"])
    p.add_argument("--workers",       type=int, default=4)
    p.add_argument("--max_per_class", type=int, default=1000)
    return p.parse_args()


def main():
    args     = parse_args()
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits       = {"train", "test"} if args.split == "all" else {args.split}
    target_lower = {c.lower() for c in TARGET_CLASSES}

    jobs, class_counts = [], {}
    with open(args.csv, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 4:
                continue
            ytid, start_str, cls, split = (
                row[0].strip(), row[1].strip(), row[2].strip(), row[3].strip()
            )
            if split not in splits or cls.lower() not in target_lower:
                continue
            if class_counts.get(cls, 0) >= args.max_per_class:
                continue
            class_counts[cls] = class_counts.get(cls, 0) + 1
            jobs.append((ytid, int(start_str), cls, split))

    log.info(f"CLIP={'on' if CLIP_AVAILABLE else 'off'}  CLAP={'on' if CLAP_AVAILABLE else 'off'}")
    log.info(f"Jobs: {len(jobs)} clips across {len(class_counts)} classes")

    ok = skip = fail = 0
    failed_log = out_dir / "failed.txt"

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_clip, ytid, start, cls, split, out_dir): (ytid, start)
            for ytid, start, cls, split in jobs
        }
        for future in as_completed(futures):
            msg = future.result()
            if msg.startswith("OK"):
                ok += 1
                log.info(msg)
            elif msg.startswith("SKIP"):
                skip += 1
            else:
                fail += 1
                log.warning(msg)
                with open(failed_log, "a") as fh:
                    fh.write(msg + "\n")

    log.info(f"Done — OK:{ok}  SKIP:{skip}  FAIL:{fail}")


if __name__ == "__main__":
    main()
