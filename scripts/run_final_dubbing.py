from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

try:
    from opencc import OpenCC
    _ZH_T2S = OpenCC("t2s")
except Exception:
    _ZH_T2S = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.infrastructure.elevenlabs_dubbing import ElevenLabsDubbingV2Adapter
from src.infrastructure.video.ffmpeg import extract_audio, mux_audio_video


def _canonical_zh(text: str) -> str:
    value = text or ""
    if _ZH_T2S is not None and re.search(r"[\u3400-\u9fff]", value):
        return _ZH_T2S.convert(value)
    return value


def _norm(text: str) -> str:
    text = _canonical_zh(unicodedata.normalize("NFKC", (text or "").lower()))
    text = re.sub(r"[^\w\u3400-\u9fff\s]", "", text, flags=re.UNICODE)
    return "".join(text.split())


def _latin_tokens(text: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{1,}", text or "")}


def _source_gate(
    reference_text: str,
    eleven_text: str,
    critical_keyterms: list[str] | None = None,
) -> dict[str, object]:
    ref = _norm(reference_text)
    got = _norm(eleven_text)
    similarity = SequenceMatcher(None, ref, got).ratio() if ref and got else 0.0
    ref_latin = _latin_tokens(reference_text)
    got_latin = _latin_tokens(eleven_text)
    unexpected_latin = sorted(got_latin - ref_latin)

    # For CJK source, a new Latin token is a high-value ASR warning (the exact
    # WiFi-vs-外卖 failure seen in the real 6s clip).
    contains_cjk = bool(re.search(r"[\u3400-\u9fff]", reference_text or ""))
    reject_unexpected_latin = contains_cjk and bool(unexpected_latin)

    normalized_terms = [_norm(term) for term in (critical_keyterms or []) if _norm(term)]
    covered_terms = [term for term in normalized_terms if term in got]
    missing_terms = [term for term in normalized_terms if term not in got]
    keyterm_coverage = len(covered_terms) / len(normalized_terms) if normalized_terms else None

    # Raw character similarity is intentionally not the only criterion for CJK:
    # Groq/Gemini may return Traditional while ElevenLabs returns Simplified, and
    # one recognizer may include short real utterances the other omitted. When a
    # corrected lexical anchor is present and no alien Latin entity appears, that
    # is stronger evidence than raw script-level similarity.
    critical_pass = bool(normalized_terms) and not missing_terms
    passed = (similarity >= 0.82 or critical_pass) and not reject_unexpected_latin
    return {
        "passed": passed,
        "similarity": round(similarity, 4),
        "unexpected_latin_tokens": unexpected_latin,
        "critical_keyterm_coverage": None if keyterm_coverage is None else round(keyterm_coverage, 4),
        "missing_critical_keyterms": missing_terms,
        "reference_text": reference_text,
        "elevenlabs_text": eleven_text,
    }


def _target_audio_gate(target_text: str, heard_text: str) -> dict[str, object]:
    expected = _norm(target_text)
    heard = _norm(heard_text)
    similarity = SequenceMatcher(None, expected, heard).ratio() if expected and heard else 0.0
    return {
        "passed": similarity >= 0.80,
        "similarity": round(similarity, 4),
        "target_text": target_text,
        "heard_text": heard_text,
    }


def _clean_keyterm(text: str) -> str:
    value = _canonical_zh(unicodedata.normalize("NFKC", text or "")).strip()
    value = re.sub(r"[\r\n\t]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"^[\s,.;:!?，。！？；：、]+|[\s,.;:!?，。！？；：、]+$", "", value)
    return value[:50].strip()


def _build_reference(audio_path: Path, source_language: str) -> tuple[list, list[str]]:
    # Lazy import: run_e2e loads project .env as an execution concern. Keeping it
    # out of module import prevents test/library callers of _source_gate from
    # mutating process credentials just by importing this module.
    import scripts.run_e2e as e2e
    asr = e2e._build_asr_provider("auto")
    primary_segments = asr.transcribe(str(audio_path), language=source_language)
    segments = list(primary_segments)
    corrected_keyterms: list[str] = []
    corrector = e2e._build_asr_corrector("auto", "auto")
    if corrector is not None:
        try:
            corrected = corrector.correct_segments(
                segments,
                language=source_language,
                audio_path=str(audio_path),
            )
            before_by_id = {getattr(s, "segment_id", str(i)): s for i, s in enumerate(primary_segments)}
            for i, seg in enumerate(corrected):
                key = getattr(seg, "segment_id", str(i))
                before = before_by_id.get(key)
                before_text = getattr(before, "source_text", "") if before is not None else ""
                after_text = getattr(seg, "source_text", "")
                if _norm(before_text) != _norm(after_text):
                    term = _clean_keyterm(after_text)
                    cjk_count = len(re.findall(r"[\u3400-\u9fff]", term))
                    latin_changed = _latin_tokens(before_text) != _latin_tokens(after_text)
                    # Bias Dubbing only with meaningful corrections. This avoids
                    # filling keyterms with harmless Simplified/Traditional spelling
                    # changes while retaining entity repairs such as WiFi -> 外卖.
                    if term and (latin_changed or cjk_count >= 6 or len(term) >= 10):
                        if term not in corrected_keyterms:
                            corrected_keyterms.append(term)
            segments = corrected
        except Exception:
            # Fail-open exactly like AudioPipeline; the final gate still compares
            # the independent transcripts instead of inventing a correction.
            pass

    # If correction did not change anything, still provide a small set of longer
    # source phrases. Dubbing v2 keyterms bias transcription/translation and are
    # especially useful for short CJK utterances with lexical ambiguity.
    if not corrected_keyterms:
        for seg in segments:
            term = _clean_keyterm(getattr(seg, "source_text", ""))
            cjk_count = len(re.findall(r"[\u3400-\u9fff]", term))
            if (cjk_count >= 4 or len(term) >= 8) and term not in corrected_keyterms:
                corrected_keyterms.append(term)
            if len(corrected_keyterms) >= 32:
                break
    return segments, corrected_keyterms[:1000]


def _fallback_mediaflow(input_path: Path, output_path: Path, source_language: str, target_language: str) -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_e2e.py"),
        "--input", str(input_path),
        "--output", str(output_path),
        "--preset", "dub-audio",
        "--source-language", source_language,
        "--target-language", target_language,
        "--tts-provider", "auto",
        "--overwrite",
    ]
    subprocess.run(command, cwd=str(ROOT), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="MediaFlow final-quality dubbing router")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-language", default="zh")
    parser.add_argument("--target-language", default="vi")
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    source = Path(args.input).resolve()
    output = Path(args.output).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    output.parent.mkdir(parents=True, exist_ok=True)

    work = output.parent / f".{output.stem}_quality_work"
    work.mkdir(parents=True, exist_ok=True)
    reference_wav = work / "reference.wav"
    eleven_audio = work / "elevenlabs_dub.flac"

    extract_audio(str(source), str(reference_wav))
    reference_segments, dubbing_keyterms = _build_reference(reference_wav, args.source_language)
    reference_text = " ".join(s.source_text for s in reference_segments).strip()

    adapter = ElevenLabsDubbingV2Adapter()
    result = adapter.dub_file(
        str(source),
        target_language=args.target_language,
        source_language=args.source_language if args.source_language != "auto" else None,
        output_audio_path=str(eleven_audio),
        reference="MediaFlow final quality router",
        keyterms=dubbing_keyterms,
    )
    source_payload = adapter.get_source_transcript(result.project_id, key_index=result.key_index)
    target_payload = adapter.get_target_transcript(
        result.project_id,
        result.language_id,
        key_index=result.key_index,
    )
    eleven_source_segments = list(source_payload.get("segments") or [])
    eleven_source_text = " ".join(str(s.get("text") or "") for s in eleven_source_segments).strip()
    gate = _source_gate(reference_text, eleven_source_text, dubbing_keyterms)

    target_segments = list(target_payload.get("segments") or [])
    eleven_target_text = " ".join(str(s.get("translation") or "") for s in target_segments).strip()
    import scripts.run_e2e as e2e
    validation_asr = e2e._build_asr_provider("auto")
    validation_segments = validation_asr.transcribe(str(eleven_audio), language=args.target_language)
    heard_target_text = " ".join(s.source_text for s in validation_segments).strip()
    audio_gate = _target_audio_gate(eleven_target_text, heard_target_text)

    speakers = sorted({str(s.get("speaker_id") or "") for s in eleven_source_segments if s.get("speaker_id")})
    report = {
        "engine_attempted": "elevenlabs-dubbing-v2",
        "project_id": result.project_id,
        "language_id": result.language_id,
        "selected_key_index": result.key_index,
        "speaker_count": len(speakers),
        "speakers": speakers,
        "source_gate": gate,
        "target_audio_gate": audio_gate,
        "dubbing_keyterms": dubbing_keyterms,
        "project_warnings": result.project_warnings,
        "language_warnings": result.language_warnings,
        "target_segments": [
            {
                "speaker_id": s.get("speaker_id"),
                "start_s": s.get("start_s"),
                "end_s": s.get("end_s"),
                "translation": s.get("translation"),
            }
            for s in target_segments
        ],
    }

    if bool(gate["passed"]) and bool(audio_gate["passed"]):
        mux_audio_video(str(source), str(eleven_audio), str(output))
        report["selected_engine"] = "elevenlabs-dubbing-v2"
        report["fallback_reason"] = None
    else:
        _fallback_mediaflow(source, output, args.source_language, args.target_language)
        report["selected_engine"] = "mediaflow-fallback"
        failed = []
        if not bool(gate["passed"]):
            failed.append("source semantic gate")
        if not bool(audio_gate["passed"]):
            failed.append("dubbed audio fidelity gate")
        report["fallback_reason"] = "ElevenLabs failed " + " + ".join(failed)

    report_path = Path(args.report).resolve() if args.report else output.with_suffix(".quality.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "report": str(report_path),
        "selected_engine": report["selected_engine"],
        "speaker_count": report["speaker_count"],
        "source_gate": gate,
        "target_audio_gate": audio_gate,
    }, ensure_ascii=True))


if __name__ == "__main__":
    main()
