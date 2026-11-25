"""yt-dlpの実行物の署名検証・検証付き更新・保留・戻し。外部へは接続しない。"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from lss.acquisition import openpgp, provider

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "yt-dlp-2026.08.19"
MODULE = (sys.executable, "-m", "yt_dlp")


# --- 公式の署名で検証する ----------------------------------------------------

@pytest.fixture
def official_release_signature():
    sums_path = FIXTURES / "SHA2-256SUMS"
    signature_path = FIXTURES / "SHA2-256SUMS.sig"
    missing = [path.relative_to(ROOT).as_posix()
               for path in (sums_path, signature_path) if not path.is_file()]
    if missing:
        pytest.skip("missing fixtures: " + ", ".join(missing))
    return sums_path.read_bytes(), signature_path.read_bytes()


def test_official_release_checksums_verify_with_the_pinned_key(official_release_signature):
    sums, signature = official_release_signature
    signer = openpgp.verify_detached(sums, signature, provider.pinned_keys())
    assert signer.fingerprint in provider.PINNED_FINGERPRINTS
    assert provider.parse_sums(sums.decode())["yt-dlp.exe"] == \
        "66674953fe251b89f4d08c5f0e35e0728679bd67ab3d7d05c0562af101dd3e7a"


def test_tampered_checksums_or_other_keys_are_rejected(official_release_signature):
    sums, signature = official_release_signature
    with pytest.raises(openpgp.SignatureError):
        openpgp.verify_detached(sums.replace(b"66674953", b"00000000"), signature, provider.pinned_keys())
    other = _TestKey()
    with pytest.raises(openpgp.SignatureError):
        openpgp.verify_detached(sums, signature, [other.public])
    with pytest.raises(openpgp.SignatureError):
        openpgp.verify_detached(sums, signature[:-10], provider.pinned_keys())


# --- 試験用の署名鍵 ------------------------------------------------------------

def _packet(tag: int, body: bytes) -> bytes:
    return bytes([0xC0 | tag, 0xFF]) + len(body).to_bytes(4, "big") + body


def _mpi(value: int) -> bytes:
    return value.bit_length().to_bytes(2, "big") + value.to_bytes((value.bit_length() + 7) // 8, "big")


class _TestKey:
    def __init__(self) -> None:
        from Cryptodome.PublicKey import RSA

        key = RSA.generate(2048)
        self._n, self._e, self._d = key.n, key.e, key.d
        material = bytes([4]) + (1_700_000_000).to_bytes(4, "big") + bytes([1]) + _mpi(key.n) + _mpi(key.e)
        self.packet = _packet(6, material)
        fingerprint = hashlib.sha1(b"\x99" + len(material).to_bytes(2, "big") + material).hexdigest().upper()
        self.public = openpgp.RsaKey(fingerprint, key.n, key.e)

    def sign(self, document: bytes) -> bytes:
        hashed = bytes([22, 33, 4]) + bytes.fromhex(self.public.fingerprint) + bytes([5, 2]) + (1).to_bytes(4, "big")
        head = bytes([4, 0x00, 1, 8]) + len(hashed).to_bytes(2, "big") + hashed
        digest = hashlib.sha256(document + head + b"\x04\xff" + len(head).to_bytes(4, "big")).digest()
        size = (self._n.bit_length() + 7) // 8
        info = bytes.fromhex("3031300d060960864801650304020105000420") + digest
        encoded = b"\x00\x01" + b"\xff" * (size - len(info) - 3) + b"\x00" + info
        value = pow(int.from_bytes(encoded, "big"), self._d, self._n)
        return _packet(2, head + (0).to_bytes(2, "big") + digest[:2] + _mpi(value))


def test_the_minimal_verifier_accepts_a_correct_v4_rsa_signature():
    key = _TestKey()
    assert openpgp.verify_detached(b"document", key.sign(b"document"), [key.public]) == key.public
    with pytest.raises(openpgp.SignatureError):
        openpgp.verify_detached(b"documenT", key.sign(b"document"), [key.public])


# --- 更新の手順 ----------------------------------------------------------------

class Release:
    """GitHubのリリースを模した配布物。"""

    def __init__(self, key: _TestKey, version: str, exe: bytes, *, sums_for: bytes | None = None,
                 signature: bytes | None = None) -> None:
        digest = hashlib.sha256(sums_for if sums_for is not None else exe).hexdigest()
        sums = f"{digest}  yt-dlp.exe\n{'0' * 64}  yt-dlp_linux\n".encode()
        base = f"https://example.test/{version}/"
        self.files = {
            provider.RELEASE_API: json.dumps({"tag_name": version, "assets": [
                {"name": "yt-dlp.exe", "browser_download_url": base + "yt-dlp.exe",
                 "digest": "sha256:" + hashlib.sha256(exe).hexdigest()},
                {"name": "SHA2-256SUMS", "browser_download_url": base + "SHA2-256SUMS"},
                {"name": "SHA2-256SUMS.sig", "browser_download_url": base + "SHA2-256SUMS.sig"},
            ]}).encode(),
            base + "yt-dlp.exe": exe,
            base + "SHA2-256SUMS": sums,
            base + "SHA2-256SUMS.sig": signature if signature is not None else key.sign(sums),
        }
        self.fetched: list[str] = []

    def __call__(self, url: str) -> bytes:
        self.fetched.append(url)
        return self.files[url]


@pytest.fixture
def signing(tmp_path, monkeypatch):
    monkeypatch.setenv("LSS_STATE_DIR", str(tmp_path / "state"))
    key = _TestKey()
    monkeypatch.setattr(provider, "pinned_keys", lambda: [key.public])
    return key


def _accept(checked: list):
    def validate(command, version):
        checked.append((Path(command[0]).read_bytes(), version))
    return validate


def test_verified_release_is_installed_and_the_previous_one_is_kept(signing):
    checked: list = []
    first = provider.check_and_install(fetch=Release(signing, "2026.09.01", b"exe-1"), validate=_accept(checked), now=100)
    assert first["status"] == "installed" and checked == [(b"exe-1", "2026.09.01")]
    current = provider.active()
    assert current.kind == "release" and current.version == "2026.09.01"
    assert Path(current.command[0]).read_bytes() == b"exe-1"

    again = provider.check_and_install(fetch=Release(signing, "2026.09.01", b"exe-1"), validate=_accept(checked), now=200)
    assert again["status"] == "current" and len(checked) == 1

    provider.check_and_install(fetch=Release(signing, "2026.09.10", b"exe-2"), validate=_accept(checked), now=300)
    provider.check_and_install(fetch=Release(signing, "2026.09.20", b"exe-3"), validate=_accept(checked), now=400)
    state = provider.load_state()
    assert state["active"]["version"] == "2026.09.20" and state["previous"]["version"] == "2026.09.10"
    assert sorted(p.name for p in (provider.root() / "versions").iterdir()) == ["2026.09.10", "2026.09.20"]

    # 有効版が壊れたら、直前の版で取得を続ける。
    Path(provider.active().command[0]).write_bytes(b"broken")
    assert provider.active().version == "2026.09.10"


@pytest.mark.parametrize("case", ["bad_signature", "checksum_mismatch", "validation_fails", "other_key"])
def test_unverifiable_or_failing_candidates_are_held_without_activation(signing, case):
    checked: list = []
    provider.check_and_install(fetch=Release(signing, "2026.09.01", b"good"), validate=_accept(checked), now=1)

    validate = _accept(checked)
    if case == "bad_signature":
        release = Release(signing, "2026.09.02", b"new", signature=signing.sign(b"something else"))
    elif case == "checksum_mismatch":
        release = Release(signing, "2026.09.02", b"new", sums_for=b"other bytes")
    elif case == "other_key":
        release = Release(_TestKey(), "2026.09.02", b"new")
    else:
        release = Release(signing, "2026.09.02", b"new")

        def validate(command, version):
            raise RuntimeError("smoke download failed")

    result = provider.check_and_install(fetch=release, validate=validate, now=2)
    assert result["status"] == "held" and result["latest"] == "2026.09.02"
    assert provider.active().version == "2026.09.01"
    assert provider.load_state()["held"]["status"] == "held"
    assert not (provider.root() / "versions" / "2026.09.02").exists()
    assert not (provider.root() / "staging").exists()
    if case != "validation_fails":
        assert len(checked) == 1  # 検証を通らない候補は実行もしない


def test_update_checks_are_daily_and_hourly_after_failures(signing, monkeypatch):
    calls: list = []
    monkeypatch.setattr(provider, "check_and_install", lambda **kw: calls.append(kw["now"]) or {"status": "current"})
    provider._save_state({})
    assert provider.ensure(now=10) is not None
    provider._save_state({"last_check": 10})
    assert provider.ensure(now=10 + 3600) is None
    assert provider.ensure(now=10 + 3600, after_failure=True) is not None
    assert provider.ensure(now=10 + 24 * 3600) is not None
    assert calls == [10, 3610, 86410]


def test_missing_provider_in_the_packaged_app_is_installed_before_acquiring(signing, monkeypatch):
    monkeypatch.setattr(provider, "is_frozen", lambda: True)
    with pytest.raises(provider.AcquisitionFailure):
        provider.active()
    checked: list = []
    result = provider.ensure(fetch=Release(signing, "2026.09.01", b"exe"), validate=_accept(checked), now=5)
    assert result["status"] == "installed" and provider.active().kind == "release"


def test_candidate_validation_runs_version_and_a_local_download():
    import yt_dlp.version

    provider.validate_candidate(MODULE, yt_dlp.version.__version__)
    with pytest.raises(RuntimeError):
        provider.validate_candidate(MODULE, "1999.01.01")
