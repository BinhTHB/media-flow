"""FFmpeg implementation of :class:`VideoVariationPort`.

Builds one ``-vf``/``-af`` filter chain from a ``VariationProfile`` and runs a
single re-encode. One pass matters: chaining separate ffmpeg invocations would
re-encode (and degrade) the video once per knob.

The adapter is intentionally dumb — it trusts the profile's bounds, which
``VariationProfile`` already validated at construction. Keeping validation in
the model means every caller, not just this adapter, is protected.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import List, Tuple

from src.core.contracts import VideoVariationPort
from src.core.models.variation import VariationProfile

logger = logging.getLogger(__name__)


class FFmpegVariationAdapter(VideoVariationPort):
    def __init__(self, *, crf: int = 20, preset: str = "fast", audio_bitrate: str = "192k") -> None:
        self._crf = int(crf)
        self._preset = preset
        self._audio_bitrate = audio_bitrate

    # -- filter construction (pure, unit-tested) ----------------------------

    @staticmethod
    def build_filters(
        profile: VariationProfile,
        *,
        frame_width: int | None = None,
        frame_height: int | None = None,
    ) -> Tuple[str, str]:
        """Return ``(video_filter, audio_filter)`` strings for the profile.

        ``frame_width``/``frame_height`` are the source dimensions. Supplying
        them makes the crop+restore path exact; without them the restore falls
        back to a ratio-based upscale that can drift by a couple of pixels.

        An empty string means "no filter for that stream".
        """
        vf: List[str] = []

        if profile.crop_ratio > 0:
            # Crop symmetrically, then restore the ORIGINAL frame size so the
            # upload keeps the source resolution and aspect ratio. Every
            # dimension is rounded to even pixels: libx264 rejects odd sizes.
            #
            # When the source size is known we scale back to those exact numbers
            # instead of dividing by the crop factor: the latter rounds twice
            # (crop then upscale) and lands 2px short on 1920x1080.
            keep = 1.0 - 2.0 * float(profile.crop_ratio)
            vf.append(f"crop=trunc(iw*{keep:.4f}/2)*2:trunc(ih*{keep:.4f}/2)*2")
            if frame_width and frame_height:
                target_w = int(frame_width) // 2 * 2
                target_h = int(frame_height) // 2 * 2
                vf.append(f"scale={target_w}:{target_h}:flags=lanczos")
            else:
                vf.append(f"scale=trunc(iw/{keep:.4f}/2)*2:trunc(ih/{keep:.4f}/2)*2:flags=lanczos")
            vf.append("setsar=1")

        if profile.mirror:
            vf.append("hflip")

        if profile.rotate_degrees:
            # Rotate, then crop the black corners the rotation creates. The
            # result is scaled back to the source size for the same reason as
            # the crop path: keep the upload's resolution identical.
            vf.append(f"rotate={float(profile.rotate_degrees):.3f}*PI/180:fillcolor=black")
            if frame_width and frame_height:
                target_w = int(frame_width) // 2 * 2
                target_h = int(frame_height) // 2 * 2
                vf.append(f"crop={target_w}:{target_h}")
            else:
                vf.append("crop=trunc(iw*0.97/2)*2:trunc(ih*0.97/2)*2")

        eq_parts: List[str] = []
        if profile.brightness:
            eq_parts.append(f"brightness={float(profile.brightness):.3f}")
        if profile.contrast != 1.0:
            eq_parts.append(f"contrast={float(profile.contrast):.3f}")
        if profile.saturation != 1.0:
            eq_parts.append(f"saturation={float(profile.saturation):.3f}")
        if eq_parts:
            vf.append("eq=" + ":".join(eq_parts))

        if profile.speed != 1.0:
            # setpts uses the inverse: faster playback = smaller multiplier.
            vf.append(f"setpts={1.0 / float(profile.speed):.6f}*PTS")

        af: List[str] = []
        if profile.speed != 1.0 or profile.pitch_semitones:
            # atempo accepts 0.5..2.0; our bounds are far narrower, so one stage
            # is always enough. Pitch is corrected independently via asetrate +
            # atempo so speed changes do not shift the voice.
            af.append(f"atempo={float(profile.speed):.4f}")
            if profile.pitch_semitones:
                semis = float(profile.pitch_semitones)
                ratio = 2.0 ** (semis / 12.0)
                af.append(f"asetrate=48000*{ratio:.6f}")
                af.append(f"atempo={1.0 / ratio:.6f}")
                af.append("aresample=48000")

        return ",".join(vf), ",".join(af)

    # -- execution ----------------------------------------------------------

    def apply(self, video_path: str, profile: VariationProfile, output_path: str) -> str:
        src = Path(video_path)
        if not src.is_file():
            raise FileNotFoundError(video_path)

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        if profile.is_identity:
            # Nothing to change: copy bytes rather than paying a re-encode.
            import shutil

            shutil.copy2(src, out)
            logger.info("Variation: identity profile, copied %s", src.name)
            return str(out)

        # Probe the source so the crop/rotate restore targets exact dimensions.
        frame_width = frame_height = None
        try:
            from src.infrastructure.video.ffmpeg import probe_video

            info = probe_video(str(src))
            frame_width, frame_height = int(info["width"]), int(info["height"])
        except Exception as exc:  # probe is best-effort; ratio fallback still works
            logger.debug("Variation: probe failed for %s (%s)", src.name, exc)

        vf, af = self.build_filters(
            profile, frame_width=frame_width, frame_height=frame_height
        )

        cmd: List[str] = ["ffmpeg", "-y", "-v", "error"]
        if profile.trim_start_seconds > 0:
            cmd += ["-ss", f"{float(profile.trim_start_seconds):.3f}"]
        cmd += ["-i", str(src)]
        if profile.trim_end_seconds > 0:
            cmd += ["-t", f"{float(profile.trim_end_seconds):.3f}"]

        if vf:
            cmd += ["-vf", vf]
        if af:
            cmd += ["-af", af]

        cmd += [
            "-c:v", "libx264",
            "-preset", self._preset,
            "-crf", str(self._crf),
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", self._audio_bitrate,
            "-movflags", "+faststart",
            str(out),
        ]

        logger.info("Variation: applying profile %r to %s", profile.to_dict(), src.name)
        result = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
        if result.returncode != 0:
            out.unlink(missing_ok=True)
            raise RuntimeError(
                f"ffmpeg variation failed ({result.returncode}): {result.stderr[:500]}"
            )
        if not out.is_file() or out.stat().st_size == 0:
            raise RuntimeError("ffmpeg variation produced no output")
        return str(out)
