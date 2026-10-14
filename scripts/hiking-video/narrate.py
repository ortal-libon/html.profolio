#!/usr/bin/env python3
"""Add a spoken narration track to assets/videos/hiking-muscles.mp4.

Each line is synthesized with Piper text-to-speech, placed at the start
of its scene (timings come from render.py) and muxed into the MP4. The
video stream is copied, not re-encoded; any old audio track is replaced.

Needs `pip install piper-tts` and a Piper voice model. The video uses
"Ryan (high)", CC BY-NC-SA 4.0:
https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-en-us-ryan-high.tar.gz

    python3 scripts/hiking-video/narrate.py path/to/en-us-ryan-high.onnx
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

from piper import PiperVoice, SynthesisConfig

from render import OUT_VIDEO, SCENES, TOTAL

# (scene key, delay after the scene starts in seconds, text)
NARRATION = [
    ("title", 0.4, "Which muscles work on a hike? Let's find out."),
    ("flat", 0.7, "On flat ground, every step uses your calves, quads, "
                  "hamstrings and glutes. Very nice."),
    ("uphill", 0.6, "Uphill! Your glutes are the main engine, and your quads "
                    "lift you up, one step at a time. Your legs are asking: "
                    "why did we come here?"),
    ("downhill", 0.6, "Downhill looks easy. It is a trap! Your quads brake on "
                      "every step, and that is why they hurt tomorrow."),
    ("core", 0.6, "Your abs keep you balanced, your back carries the snacks, "
                  "and your arms push on the poles."),
    ("recap", 0.4, "Great success! Glutes, quads, calves, hamstrings, shins and core. "
                   "Stretch them after the hike."),
]

LENGTH_SCALE = 0.95  # <1 speaks a little faster
GAP = 0.25  # minimum silence before the next scene's line


def synthesize(voice, text, path):
    with wave.open(str(path), "wb") as wav:
        voice.synthesize_wav(text, wav, syn_config=SynthesisConfig(length_scale=LENGTH_SCALE))
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / wav.getframerate()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", type=Path, help="Piper .onnx voice model (with its .onnx.json next to it)")
    args = ap.parse_args()

    voice = PiperVoice.load(str(args.model))
    starts = {s["key"]: s["start"] for s in SCENES}
    ends = {s["key"]: s["start"] + s["dur"] for s in SCENES}

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        clips, problems = [], []
        for i, (key, delay, text) in enumerate(NARRATION):
            path = tmp / f"line{i}.wav"
            dur = synthesize(voice, text, path)
            at = starts[key] + delay
            limit = ends[key] - GAP if key != "recap" else TOTAL - 0.9
            print(f"{key:9s} {at:5.1f}s -> {at + dur:5.1f}s (scene ends {ends[key]:.1f}s)")
            if at + dur > limit:
                problems.append(f"'{key}' line runs {at + dur - limit:.2f}s too long")
            clips.append((path, at))
        if problems:
            sys.exit("narration does not fit:\n  " + "\n  ".join(problems))

        inputs, filters = [], []
        for n, (path, at) in enumerate(clips, start=1):
            inputs += ["-i", str(path)]
            ms = int(at * 1000)
            filters.append(f"[{n}:a]aresample=48000,adelay={ms}|{ms}[a{n}]")
        mix = "".join(f"[a{n}]" for n in range(1, len(clips) + 1))
        filters.append(f"{mix}amix=inputs={len(clips)}:normalize=0,"
                       f"apad,atrim=0:{TOTAL},afade=t=out:st={TOTAL - 0.8}:d=0.8,"
                       f"loudnorm=I=-16:TP=-1.5[aout]")
        out = tmp / "out.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(OUT_VIDEO), *inputs,
                        "-filter_complex", ";".join(filters),
                        "-map", "0:v", "-map", "[aout]", "-c:v", "copy",
                        "-c:a", "aac", "-b:a", "128k", "-ac", "1", "-ar", "48000",
                        "-movflags", "+faststart", str(out)], check=True)
        shutil.move(out, OUT_VIDEO)
    print(f"added narration to {OUT_VIDEO.name}")


if __name__ == "__main__":
    main()
