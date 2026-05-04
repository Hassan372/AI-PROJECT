"""
VGGSound animal clips downloader.

For each entry in vggsound.csv whose class belongs to the 'animals' category,
downloads the 10-second segment from YouTube with yt-dlp, cuts it with ffmpeg,
and extracts the mid-frame as a JPEG.

Output layout:
  <out_dir>/
    <split>/          (train / test)
      <class>/
        <ytid>_<start6d>.mp4   — 10 s video clip
        <ytid>_<start6d>.wav   — 10 s audio (16-bit PCM, 24 kHz mono)
        <ytid>_<start6d>.jpg   — mid-frame RGB image

Usage:
  python download_vggsound.py --out_dir data/raw --workers 4 --split train
  python download_vggsound.py --out_dir data/raw --split all   # train + test
"""

import argparse
import csv
import logging
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    import imageio_ffmpeg
    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    FFMPEG = "ffmpeg"

# ── Animal classes (from stat.csv, category == "animals") ──────────────────
ANIMAL_CLASSES = {
    "alligators, crocodiles hissing",
    "baltimore oriole calling",
    "barn swallow calling",
    "bee, wasp, etc. buzzing",
    "bird chirping, tweeting",
    "bird squawking",
    "bird wings flapping",
    "black capped chickadee calling",
    "bull bellowing",
    "canary calling",
    "cat caterwauling",
    "cat growling",
    "cat hissing",
    "cat meowing",
    "cat purring",
    "cat caterwauling",
    "cattle mooing",
    "cattle, bovinae cowbell",
    "cheetah chirrup",
    "chicken clucking",
    "chicken crowing",
    "chimpanzee pant-hooting",
    "chipmunk chirping",
    "chinchilla barking",
    "cow lowing",
    "coyote howling",
    "cricket chirping",
    "crow cawing",
    "cuckoo bird calling",
    "dinosaurs bellowing",
    "dog barking",
    "dog baying",
    "dog bow-wow",
    "dog growling",
    "dog howling",
    "dog whimpering",
    "donkey, ass braying",
    "duck quacking",
    "eagle screaming",
    "elephant trumpeting",
    "elk bugling",
    "ferret dooking",
    "fly, housefly buzzing",
    "fox barking",
    "francolin calling",
    "frog croaking",
    "gibbon howling",
    "goat bleating",
    "goose honking",
    "horse clip-clop",
    "horse neighing",
    "lions growling",
    "lions roaring",
    "magpie calling",
    "mosquito buzzing",
    "mouse pattering",
    "mouse squeaking",
    "mynah bird singing",
    "otters growling",
    "otter growling",
    "owl hooting",
    "parrot talking",
    "penguins braying",
    "pheasant crowing",
    "pig oinking",
    "pigeon, dove cooing",
    "sea lion barking",
    "sheep bleating",
    "snake hissing",
    "snake rattling",
    "turkey gobbling",
    "warbler chirping",
    "whale calling",
    "wood thrush calling",
    "woodpecker pecking tree",
    "zebra braying",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def stem(ytid: str, start: int) -> str:
    return f"{ytid}_{start:06d}"


def already_done(out_dir: Path, split: str, cls: str, ytid: str, start: int) -> bool:
    s = stem(ytid, start)
    d = out_dir / split / cls
    return (d / f"{s}.mp4").exists() and (d / f"{s}.wav").exists() and (d / f"{s}.jpg").exists()


def download_clip(ytid: str, start: int, cls: str, split: str, out_dir: Path) -> str:
    """
    Download + cut one 10-second clip.  Returns a status string for logging.
    """
    s = stem(ytid, start)
    target_dir = out_dir / split / cls.replace("/", "_").replace(",", "")
    target_dir.mkdir(parents=True, exist_ok=True)

    mp4_out = target_dir / f"{s}.mp4"
    wav_out = target_dir / f"{s}.wav"
    jpg_out = target_dir / f"{s}.jpg"

    if mp4_out.exists() and wav_out.exists() and jpg_out.exists():
        return f"SKIP {s}"

    url = f"https://www.youtube.com/watch?v={ytid}"

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        raw_video = tmp / "raw.%(ext)s"

        # ── 1. Download best video+audio up to 720p ──────────────────────
        # Use a pre-merged progressive format (video+audio in one file).
        # DASH streams require EJS challenge solving to merge; format 18 / best[acodec!=none]
        # is always a progressive stream that needs no merging.
        dl_cmd = [
            "yt-dlp",
            "--quiet",
            "--no-warnings",
            "--js-runtimes", "node",
            "--ffmpeg-location", str(Path(FFMPEG).parent),
            "--format", "best[ext=mp4][acodec!=none]/best[acodec!=none]",
            "--output", str(raw_video),
            "--no-playlist",
            url,
        ]
        result = subprocess.run(dl_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return f"FAIL_DL {s}: {result.stderr.strip()[:120]}"

        # Find the downloaded file (extension may vary)
        raw_files = list(tmp.glob("raw.*"))
        if not raw_files:
            return f"FAIL_DL {s}: no file found after download"
        raw_file = raw_files[0]

        # ── 2. Cut 10-second segment → .mp4 ─────────────────────────────
        cut_cmd = [
            FFMPEG, "-y", "-loglevel", "error",
            "-ss", str(start),
            "-i", str(raw_file),
            "-t", "10",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac",
            str(mp4_out),
        ]
        r = subprocess.run(cut_cmd, capture_output=True, text=True)
        if r.returncode != 0:
            return f"FAIL_CUT {s}: {r.stderr.strip()[:120]}"

        # ── 3. Extract audio → 24 kHz mono WAV ──────────────────────────
        audio_cmd = [
            FFMPEG, "-y", "-loglevel", "error",
            "-i", str(mp4_out),
            "-vn",
            "-ar", "24000",
            "-ac", "1",
            "-sample_fmt", "s16",
            str(wav_out),
        ]
        r = subprocess.run(audio_cmd, capture_output=True, text=True)
        if r.returncode != 0:
            return f"FAIL_WAV {s}: {r.stderr.strip()[:120]}"

        # ── 4. Extract mid-frame → JPEG ──────────────────────────────────
        mid_time = 5.0  # seconds into the 10-s clip
        frame_cmd = [
            FFMPEG, "-y", "-loglevel", "error",
            "-ss", str(mid_time),
            "-i", str(mp4_out),
            "-frames:v", "1",
            "-q:v", "2",
            str(jpg_out),
        ]
        r = subprocess.run(frame_cmd, capture_output=True, text=True)
        if r.returncode != 0:
            return f"FAIL_FRAME {s}: {r.stderr.strip()[:120]}"

    return f"OK {s}"


def parse_args():
    p = argparse.ArgumentParser(description="Download VGGSound animal clips")
    p.add_argument("--csv", default="VGGSound/data/vggsound.csv",
                   help="Path to vggsound.csv")
    p.add_argument("--out_dir", default="data/raw",
                   help="Root output directory")
    p.add_argument("--split", default="train",
                   choices=["train", "test", "all"],
                   help="Which split to download")
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel download workers")
    p.add_argument("--max_per_class", type=int, default=None,
                   help="Cap clips per class (useful for a small pilot)")
    p.add_argument("--classes", nargs="*", default=None,
                   help="Restrict to these animal classes (space-separated)")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_classes = (
        {c.lower() for c in args.classes}
        if args.classes
        else {c.lower() for c in ANIMAL_CLASSES}
    )

    # ── Read CSV and filter ───────────────────────────────────────────────
    jobs = []
    class_counts: dict[str, int] = {}
    splits = {"train", "test"} if args.split == "all" else {args.split}

    with open(args.csv, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 4:
                continue
            ytid, start_str, cls, split = row[0].strip(), row[1].strip(), row[2].strip(), row[3].strip()
            if split not in splits:
                continue
            if cls.lower() not in target_classes:
                continue
            if args.max_per_class is not None:
                if class_counts.get(cls, 0) >= args.max_per_class:
                    continue
            class_counts[cls] = class_counts.get(cls, 0) + 1
            jobs.append((ytid, int(start_str), cls, split))

    log.info(f"Jobs to process: {len(jobs)} clips across {len(class_counts)} animal classes")

    # ── Dispatch ──────────────────────────────────────────────────────────
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
                if ok % 50 == 0:
                    log.info(f"Progress — OK:{ok}  SKIP:{skip}  FAIL:{fail}")
            elif msg.startswith("SKIP"):
                skip += 1
            else:
                fail += 1
                log.warning(msg)
                with open(failed_log, "a") as fh:
                    fh.write(msg + "\n")

    log.info(f"Done — OK:{ok}  SKIP:{skip}  FAIL:{fail}")
    if fail:
        log.info(f"Failed entries written to {failed_log}")


if __name__ == "__main__":
    main()
