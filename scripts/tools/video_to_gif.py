# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert video files to optimized GIFs.

Uses ffmpeg for high-quality GIF encoding with diff-based palette generation
and delta-frame encoding (only stores changed pixels per frame). Optionally
uses gifsicle for further lossy compression when --max-size is specified.

Usage:
    python video_to_gif.py video.mp4                       # medium preset
    python video_to_gif.py video.mp4 --max-size 5          # auto-fit to <=5 MB
    python video_to_gif.py video.mp4 -q high               # high quality preset
    python video_to_gif.py video.mp4 --fps 15 --width 640  # manual control

Dependencies:
    Required: ffmpeg   (apt/brew install ffmpeg, or conda install -c conda-forge ffmpeg)
    Optional: gifsicle (apt/brew install gifsicle) — enables --max-size
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

QUALITY_PRESETS = {
    "low": {"fps": 10, "width": 320, "colors": 64},
    "medium": {"fps": 15, "width": 480, "colors": 256},
    "high": {"fps": 20, "width": 640, "colors": 256},
    "original": {"fps": None, "width": None, "colors": 256},
}

MIN_WIDTH = 200


def require(tool: str, reason: str = ""):
    """Exit with an install hint if a required CLI tool is missing."""
    if shutil.which(tool):
        return
    msg = f"Error: '{tool}' not found. Install: apt install {tool} (or conda install -c conda-forge {tool})"
    if reason:
        msg += f" - needed for {reason}"
    sys.exit(msg)


def probe_video(path: str) -> tuple[int, int, float]:
    """Return (width, height, fps) of a video via ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate", "-of", "csv=p=0", path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        sys.exit(f"Error: ffprobe failed on '{path}'")
    parts = result.stdout.strip().split(",")
    width, height = int(parts[0]), int(parts[1])
    num, den = parts[2].split("/")
    return width, height, int(num) / int(den)


def render_gif(video: str, output: str, fps: int, width: int | None, colors: int) -> int:
    """Render a GIF via ffmpeg with diff-based palette and delta-frame encoding.

    Returns:
        Output file size in bytes.
    """
    vf_parts = [f"fps={fps}"]
    if width:
        vf_parts.append(f"scale={width}:-1:flags=lanczos")

    vf = ",".join(vf_parts)
    lavfi = (
        f"{vf},split[s0][s1];"
        f"[s0]palettegen=max_colors={colors}:stats_mode=diff[p];"
        f"[s1][p]paletteuse=dither=floyd_steinberg:diff_mode=rectangle"
    )

    result = subprocess.run(
        ["ffmpeg", "-y", "-i", video, "-lavfi", lavfi, "-loop", "0", output],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        sys.exit(f"Error: ffmpeg encoding failed:\n{result.stderr[-500:]}")

    return Path(output).stat().st_size


def gifsicle_optimize(src: str, dst: str, lossy: int) -> int:
    """Apply gifsicle lossy+LZW optimization. Returns output size in bytes."""
    subprocess.run(["gifsicle", "-O3", f"--lossy={lossy}", src, "-o", dst], capture_output=True)
    return Path(dst).stat().st_size


def fit_to_size(video: str, output: str, max_bytes: int,
                src_width: int, src_fps: float,
                user_fps: int | None, user_width: int | None, colors: int) -> int:
    """Auto-select width and lossy compression to fit a GIF under max_bytes.

    Renders a low-res probe to estimate bytes-per-pixel, predicts the optimal
    width, then applies gifsicle lossy compression if still over budget.
    Typically requires only 1-2 ffmpeg renders.

    Returns:
        Final output file size in bytes.
    """
    fps = user_fps or min(15, int(src_fps))

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_raw = str(Path(tmpdir) / "raw.gif")
        tmp_opt = str(Path(tmpdir) / "opt.gif")

        if user_width:
            target_width = user_width
        else:
            probe_w = min(320, src_width)
            print(f"Probing at {probe_w}px...", end=" ", flush=True)
            probe_size = render_gif(video, tmp_raw, fps, probe_w, colors)
            bytes_per_px2 = probe_size / (probe_w * probe_w)
            target_width = min(src_width, int((max_bytes / bytes_per_px2) ** 0.5))
            target_width = max(MIN_WIDTH, target_width)
            print(f"estimated {target_width}px")

        print(f"Rendering at {target_width}px @ {fps}fps...", end=" ", flush=True)
        raw_size = render_gif(video, tmp_raw, fps, target_width, colors)
        print(fmt_size(raw_size), flush=True)

        if raw_size <= max_bytes:
            shutil.copy2(tmp_raw, output)
            return raw_size

        lo, hi, best_lossy = 20, 200, -1
        for _ in range(7):
            if lo > hi:
                break
            mid = (lo + hi) // 2
            if gifsicle_optimize(tmp_raw, tmp_opt, mid) <= max_bytes:
                best_lossy = mid
                hi = mid - 1
            else:
                lo = mid + 1

        if best_lossy >= 0:
            final_size = gifsicle_optimize(tmp_raw, output, best_lossy)
            print(f"Optimized: lossy={best_lossy}, {fmt_size(final_size)}")
            return final_size

        # Probe estimate was off — scale down proportionally and retry once
        scale = (max_bytes / raw_size) ** 0.5
        fallback_w = max(MIN_WIDTH, int(target_width * scale * 0.95))
        print(f"Adjusting to {fallback_w}px...", end=" ", flush=True)
        raw_size = render_gif(video, tmp_raw, fps, fallback_w, colors)
        print(fmt_size(raw_size), flush=True)

        if raw_size <= max_bytes:
            shutil.copy2(tmp_raw, output)
            return raw_size

        final_size = gifsicle_optimize(tmp_raw, output, 200)
        print(f"Optimized: lossy=200, {fmt_size(final_size)}")
        return final_size


def fmt_size(n_bytes: int | float) -> str:
    """Format a byte count as a human-readable string (e.g. 4.9 MB)."""
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Convert video to optimized GIF.")
    parser.add_argument("video", help="Path to the input video file.")
    parser.add_argument("-o", "--output", help="Output GIF path (default: same folder as video).")
    parser.add_argument(
        "-q", "--quality", choices=QUALITY_PRESETS.keys(), default="medium",
        help="Quality preset (default: medium). Individual flags override preset values.",
    )
    parser.add_argument("--fps", type=int, help="Output frames per second.")
    parser.add_argument("--width", type=int, help="Output width in pixels (height scales proportionally).")
    parser.add_argument("--colors", type=int, choices=[2**i for i in range(1, 9)],
                        help="Max colors in palette (power of 2, max 256).")
    parser.add_argument("--max-size", type=float, metavar="MB",
                        help="Target max file size in MB. Auto-selects width and compression to fit.")
    return parser.parse_args()


def main():
    """Entry point for video-to-GIF conversion."""
    args = parse_args()

    video_path = Path(args.video)
    if not video_path.is_file():
        sys.exit(f"Error: file not found '{video_path}'")

    output_path = args.output or str(video_path.with_suffix(".gif"))
    preset = QUALITY_PRESETS[args.quality]
    require("ffmpeg")

    src_width, src_height, src_fps = probe_video(str(video_path))
    print(f"Source:     {video_path} ({src_width}x{src_height}, {src_fps:.1f}fps)")

    if args.max_size is not None:
        require("gifsicle", "--max-size")
        colors = args.colors or 256
        max_bytes = int(args.max_size * 1024 * 1024)
        print(f"Target: <= {args.max_size:.1f} MB, colors={colors}")

        size = fit_to_size(
            str(video_path), output_path, max_bytes,
            src_width, src_fps, args.fps, args.width, colors,
        )
    else:
        fps = args.fps or preset["fps"] or int(src_fps)
        width = args.width or preset["width"]
        colors = args.colors or preset["colors"]

        print(f"Settings: fps={fps}, width={width or 'source'}, colors={colors}")
        size = render_gif(str(video_path), output_path, fps, width, colors)

        if shutil.which("gifsicle"):
            opt_path = output_path + ".opt"
            opt_size = gifsicle_optimize(output_path, opt_path, 30)
            if opt_size < size:
                shutil.move(opt_path, output_path)
                print(f"Optimized: {fmt_size(size)} -> {fmt_size(opt_size)}")
                size = opt_size
            else:
                Path(opt_path).unlink(missing_ok=True)

    print(f"Saved:      {output_path} ({fmt_size(size)})")


if __name__ == "__main__":
    main()
