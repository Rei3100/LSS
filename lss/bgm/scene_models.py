"""Local GPU models for scene names: Japanese dialogue transcription and the order-sheet/name language model.

Pinned upstream revisions; weights are fetched once into the LSS model store (``<models>/scene-names/``) and
then used offline. Safety (a 2026-10-16 prototype silently spilled into shared system memory for two hours):
GPU memory is capped so an overflow fails instead of crawling, every generation has a time limit, the models
are loaded one at a time and released after use, and the GPU is leased like the separation workers.

Measured on the 9-episode corpus-a evaluation set (`release-evidence/scene-naming-20261016/`): transcription of
the source mix beats the separated speech stem; Qwen3-14B 4-bit without thinking, one call per cue, ~10 s per cue.
"""
from __future__ import annotations

import functools
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

ASR = {"repository": "kotoba-tech/kotoba-whisper-v2.0", "revision": "7eb575277d18909a4af8a24e3ae8cce2e99794ae",
       "license": "Apache-2.0", "chunk_seconds": 15, "batch": 8}
LLM = {"repository": "Qwen/Qwen3-14B", "revision": "40c069824f4251a91eefaf281ebe4c544efd3e18",
       "license": "Apache-2.0", "quantization": "bitsandbytes-nf4-bf16-compute", "thinking": False,
       "max_new_tokens": 260, "max_seconds": 90}
MEMORY_FRACTION = 0.85
PATTERNS = ["*.json", "*.safetensors", "*.txt", "tokenizer*", "*.model", "*.jinja"]


class SceneModelsError(RuntimeError):
    pass


def model_root() -> Path:
    """``LSS_SCENE_MODEL_DIR`` places these large weights on another drive (the separation models stay put)."""
    import os

    override = os.environ.get("LSS_SCENE_MODEL_DIR")
    if override:
        return Path(override)
    from lss.core import models_store

    return Path(models_store.cache_dir()) / "scene-names"


def job_execution_profile() -> str:
    """Internal scene-job opt-in; ordinary runs keep the legacy cadence."""
    import os

    from lss.bgm.scene_execution import validate_profile

    return validate_profile(os.environ.get("LSS_SCENE_EXECUTION_PROFILE", "legacy"))


def ensure(spec: dict) -> Path:
    """The pinned snapshot in the model store, fetched once (no symlinks: Windows without the privilege)."""
    target = model_root() / spec["repository"].replace("/", "--") / spec["revision"]
    marker = target / ".lss-complete.json"
    if marker.is_file():
        return target
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SceneModelsError("場面名のモデルを取得する部品がありません") from exc
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(spec["repository"], revision=spec["revision"], local_dir=str(target), allow_patterns=PATTERNS)
    marker.write_text(json.dumps({"repository": spec["repository"], "revision": spec["revision"],
                                  "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S")}), encoding="utf-8")
    return target


def _merge(windows: list[tuple[float, float]], gap: float = 2.0) -> list[tuple[float, float]]:
    merged = []
    for start, end in sorted(windows):
        if merged and start <= merged[-1][1] + gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _file_stat_identity(path: Path) -> tuple[int | None, int | None, int, int, int]:
    stat = Path(path).stat()
    return (getattr(stat, "st_dev", None), getattr(stat, "st_ino", None), stat.st_size,
            stat.st_mtime_ns, stat.st_ctime_ns)


@functools.lru_cache(maxsize=8)
def _ffmpeg_identity(path: str, file_identity: tuple) -> dict:
    """Cache an executable binding by resolved path and file identity; never persist input audio bytes."""
    executable = Path(path).resolve()
    before = _file_stat_identity(executable)
    if before != file_identity:
        return {"path": str(executable), "stat_at_binding": list(file_identity),
                "stat_before_hash": list(before), "identity_status": "unverified_changed_before_binding"}
    digest = hashlib.sha256()
    with executable.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    version = subprocess.run([str(executable), "-version"], capture_output=True, check=True, timeout=30,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
    first_line = version.decode("utf-8", errors="replace").splitlines()
    after = _file_stat_identity(executable)
    stable = before == after
    return {"path": str(executable), "stat_at_binding": list(file_identity),
            "stat_after_binding": list(after), "identity_status": "unchanged_at_binding_checks" if stable else
            "unverified_changed_during_binding", "binary_sha256": digest.hexdigest() if stable else None,
            "version": first_line[0] if stable and first_line else None,
            "version_output_sha256": hashlib.sha256(version).hexdigest() if stable else None}


def _runtime_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


class SceneModels:
    """Lazy, one-at-a-time local models; ``release()`` frees the GPU (and the lease)."""

    def __init__(self, device: str = "cuda:0", execution_profile: str = "legacy"):
        from lss.bgm.scene_execution import SceneCadence

        self.device = device
        self.execution_profile = execution_profile
        self._cadence = SceneCadence(execution_profile)
        self._asr = self._llm = self._tok = None
        self._lease = None
        self.last_transcription_receipt = None

    def revision(self) -> dict:
        return {"asr": {k: ASR[k] for k in ("repository", "revision")},
                "llm": {k: LLM[k] for k in ("repository", "revision", "quantization", "thinking")}}

    def _gpu(self):
        if self._lease is None:
            from lss.core import gpu_lease

            # Never run scene inference without the same cross-process lease used by the main worker.
            self._lease = gpu_lease.acquire()
        import torch

        try:
            torch.cuda.set_per_process_memory_fraction(MEMORY_FRACTION, 0)
        except Exception:
            try:
                self.release()
            except Exception:  # noqa: BLE001 - preserve the lease/configuration failure
                pass
            raise

    def _prepare_model(self, name: str, spec: dict) -> str | None:
        """Fetch weights without a GPU lease; hold the lease only while loading/running a model."""
        if getattr(self, name) is not None:
            self._gpu()
            return None
        if self._lease is not None or self._asr is not None or self._llm is not None or self._tok is not None:
            self.release()
        path = ensure(spec)
        try:
            self._gpu()
        except Exception:
            try:
                self.release()
            except Exception:  # noqa: BLE001 - preserve the lease acquisition/configuration failure
                pass
            raise
        return str(path)

    def _drop(self, *names):
        import gc

        import torch

        for name in names:
            setattr(self, name, None)
        gc.collect()
        torch.cuda.empty_cache()

    def transcribe(self, source: Path, windows: list[tuple[float, float]]) -> list[dict]:
        """Dialogue lines with source times for the given windows only (the source mix, 16 kHz mono)."""
        import torch
        from transformers import pipeline

        if self._asr is None:
            path = self._prepare_model("_asr", ASR)
            try:
                self._asr = pipeline("automatic-speech-recognition", model=path, dtype=torch.float16,
                                     device=self.device)
            except Exception:
                try:
                    self.release()
                except Exception:  # noqa: BLE001 - preserve the model-load failure
                    pass
                raise
        else:
            self._gpu()
        from lss.paths import is_frozen, resource_root

        ffmpeg = resource_root() / "bin" / "ffmpeg.exe"
        if not ffmpeg.is_file() and not is_frozen():
            # Source development may use a PATH installation when its optional bundled binary is absent.
            ffmpeg = Path("ffmpeg")
        executable = str(ffmpeg)
        if ffmpeg.name == "ffmpeg" and not ffmpeg.is_file():
            import shutil

            executable = shutil.which(executable) or executable
        executable = str(Path(executable).resolve())
        ffmpeg_stat = _file_stat_identity(Path(executable))
        ffmpeg_identity = dict(_ffmpeg_identity(executable, ffmpeg_stat))
        input_windows = []
        self.last_transcription_receipt = None
        lines = []
        for window_id, (start, end) in enumerate(_merge(windows)):
            argv = [executable, "-v", "error", "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", str(source),
                    "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "-"]
            raw = subprocess.run(argv, capture_output=True, check=True,
                                 timeout=600, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
            audio = np.frombuffer(raw, dtype=np.float32).copy()
            window_receipt = {"window_id": window_id, "source_start_seconds": start, "source_end_seconds": end,
                              "ffmpeg_argv": argv,
                              "input": {"sha256": hashlib.sha256(raw).hexdigest(), "byte_length": len(raw),
                                        "sample_count": int(audio.size), "dtype": "float32-le",
                                        "sampling_rate": 16000, "accepted_by_existing_minimum": audio.size >= 1600}}
            input_windows.append(window_receipt)
            if audio.size < 1600:
                continue
            result = self._asr({"raw": audio, "sampling_rate": 16000}, chunk_length_s=ASR["chunk_seconds"],
                               batch_size=ASR["batch"], return_timestamps=True,
                               generate_kwargs={"language": "ja", "task": "transcribe"})
            self._rest_after_success("asr_pipeline")
            for chunk in result.get("chunks") or []:
                a, b = chunk["timestamp"]
                text = (chunk.get("text") or "").strip()
                if a is None or not text:
                    continue
                lines.append({"start": round(start + a, 2), "end": round(start + (b if b is not None else a + 2), 2),
                              "text": text})
        result = _drop_repeats(lines)
        ffmpeg_identity["binary_sha256_at_binding"] = ffmpeg_identity.get("binary_sha256")
        try:
            ffmpeg_after_decode = _file_stat_identity(Path(executable))
        except OSError:
            ffmpeg_after_decode = None
        ffmpeg_identity["stat_after_decode"] = list(ffmpeg_after_decode) if ffmpeg_after_decode else None
        if ffmpeg_identity.get("identity_status") == "unchanged_at_binding_checks":
            ffmpeg_identity["identity_status"] = ("unchanged_at_observation_points"
                                                   if ffmpeg_after_decode == ffmpeg_stat else
                                                   "unverified_changed_during_decode")
        if ffmpeg_identity["identity_status"] != "unchanged_at_observation_points":
            ffmpeg_identity["binary_sha256"] = None
            ffmpeg_identity["version"] = None
            ffmpeg_identity["version_output_sha256"] = None
        self.last_transcription_receipt = {
            "schema": "scene-asr-input-lineage-r1", "source_path": str(Path(source).resolve()),
            "ffmpeg": ffmpeg_identity,
            "asr": {"revision": {k: ASR[k] for k in ("repository", "revision", "chunk_seconds", "batch")},
                    "dtype": "torch.float16", "device": self.device,
                    "pipeline": "automatic-speech-recognition", "timestamps": True,
                    "generate_kwargs": {"language": "ja", "task": "transcribe"},
                    "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                                "packages": {name: _runtime_version(name) for name in
                                             ("torch", "transformers", "tokenizers", "accelerate",
                                              "huggingface-hub")},
                                "torch_cuda": getattr(getattr(torch, "version", None), "cuda", None)}},
            "windows": input_windows,
            "execution_cadence": self.cadence_receipt(),
            "transcript_sha256": hashlib.sha256(json.dumps(result, ensure_ascii=False, sort_keys=True,
                                                            separators=(",", ":")).encode("utf-8")).hexdigest(),
        }
        return result

    def _load_llm(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if self._llm is None:
            path = self._prepare_model("_llm", LLM)
            try:
                quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                           bnb_4bit_compute_dtype=torch.bfloat16)
                self._tok = AutoTokenizer.from_pretrained(path)
                self._llm = AutoModelForCausalLM.from_pretrained(path, quantization_config=quant,
                                                                 device_map=self.device)
            except Exception:
                try:
                    self.release()
                except Exception:  # noqa: BLE001 - preserve the model-load failure
                    pass
                raise
        else:
            self._gpu()

    def _ask(self, prompt: str, max_new: int) -> str:
        import torch

        self._load_llm()
        ids = self._tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                            return_tensors="pt", enable_thinking=LLM["thinking"])
        # transformers >= 5 returns a BatchEncoding here (4.x returned the id tensor itself).
        ids = (ids["input_ids"] if hasattr(ids, "keys") else ids).to(self._llm.device)
        try:
            with torch.inference_mode():
                out = self._llm.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new, do_sample=False,
                                         temperature=None, top_p=None, top_k=None, max_time=LLM["max_seconds"])
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise SceneModelsError("GPUのメモリが足りません") from exc
        self._rest_after_success("llm_generate")
        return self._tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

    def _rest_after_success(self, operation: str):
        if self._cadence.rest_seconds <= 0:
            return
        import torch

        self._cadence.after_success(operation, torch.cuda.synchronize, torch.cuda.empty_cache)

    def cadence_receipt(self) -> dict:
        return {"profile": self.execution_profile, "rest_seconds": self._cadence.rest_seconds,
                "events": [dict(event) for event in self._cadence.events]}

    def order(self, prompt: str) -> dict:
        answer = self._ask(prompt, LLM["max_new_tokens"])
        try:
            parsed = json.loads(answer[answer.index("{"): answer.rindex("}") + 1])
        except ValueError as exc:
            raise SceneModelsError("名付けの出力を読めません") from exc
        fields = ("use", "direction", "category", "name", "evidence", "name_basis")
        result = {k: (parsed.get(k) if isinstance(parsed.get(k), str) and parsed.get(k).strip()
                      and parsed.get(k) != "null" else None) for k in fields}
        if result["name_basis"] not in ("scene", "context_music"):
            result["name_basis"] = None
        return result

    def prefix(self, work_title: str, evidence: str = "") -> str:
        """``evidence``: what LSS itself holds or fetched about the series (soundtrack album titles in the
        user's reference library, the official site's opening text, the file's description)."""
        answer = self._ask(
            f"アニメ作品「{work_title}」の、このシリーズのBGMの番号（Mナンバー）に付ける英字の略号を2文字で答えてください。\n"
            f"このシリーズについて手元にある情報:\n{evidence[:1500]}\n\n"
            "同じ題名の別シリーズと区別できる略号にする。サウンドトラックや番組のシリーズ副題（題名の後の「～…～」など）が"
            "あればその語の頭文字を使う（例: 「わんだふるぷりきゅあ！」→WF、「キミとアイドルプリキュア♪」→KT）。"
            "題名全体の頭文字は別シリーズと同じになるので使わない。"
            '出力はJSONのみ: {"prefix": "XX", "series_subtitle": "根拠にした副題"}', 60)
        try:
            return json.loads(answer[answer.index("{"): answer.rindex("}") + 1])["prefix"]
        except (ValueError, KeyError) as exc:
            raise SceneModelsError("略号の出力を読めません") from exc

    def release(self):
        try:
            self._drop("_asr", "_llm", "_tok")
        finally:
            if self._lease:
                try:
                    self._lease.close()
                except Exception:  # noqa: BLE001
                    pass
            self._lease = None


STOCK = ("ご視聴ありがとうございました", "チャンネル登録", "お疲れ様でした")


def _drop_repeats(lines: list[dict]) -> list[dict]:
    """Whisper repeats and stock phrases on music-only stretches; keep the first two of a run."""
    kept, previous, run = [], None, 0
    for line in lines:
        if any(s in line["text"] for s in STOCK):
            continue
        run = run + 1 if line["text"] == previous else 0
        previous = line["text"]
        if run < 2:
            kept.append(line)
    return kept
