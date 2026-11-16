"""取得ツールの配布物に付く、OpenPGPの分離署名（v4・RSA）を検証する。

外部のgpgや暗号ライブラリに依存しない。対応するのは、yt-dlpの公開鍵と署名の形式
（v4公開鍵・副鍵、RSA、SHA-256/384/512、バイナリ文書の署名）だけで、それ以外は拒否する。
信頼の根拠は、リポジトリに固定した公開鍵と指紋であり、署名に含まれる情報ではない。
"""
from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass

_RSA_ALGORITHMS = frozenset({1, 3})  # RSA (Encrypt or Sign), RSA Sign-Only
_HASHES = {
    8: ("sha256", bytes.fromhex("3031300d060960864801650304020105000420")),
    9: ("sha384", bytes.fromhex("3041300d060960864801650304020205000430")),
    10: ("sha512", bytes.fromhex("3051300d060960864801650304020305000440")),
}
_BINARY_DOCUMENT = 0x00


class SignatureError(ValueError):
    """署名が検証できない。形式が対応外のときも同じ扱い（通さない）。"""


@dataclass(frozen=True)
class RsaKey:
    fingerprint: str
    n: int
    e: int

    @property
    def key_id(self) -> str:
        return self.fingerprint[-16:]


def dearmor(text: str) -> bytes:
    lines = [line.strip() for line in text.strip().splitlines()]
    try:
        start = next(i for i, line in enumerate(lines) if line.startswith("-----BEGIN PGP"))
        end = next(i for i, line in enumerate(lines) if line.startswith("-----END PGP"))
    except StopIteration as exc:
        raise SignatureError("ASCII armorの境界がありません") from exc
    body = lines[start + 1:end]
    while body and body[0]:  # ヘッダ行（Version: 等）
        body.pop(0)
    data = "".join(line for line in body if line and not line.startswith("="))
    try:
        return base64.b64decode(data, validate=True)
    except ValueError as exc:
        raise SignatureError("ASCII armorを復号できません") from exc


def packets(data: bytes):
    """(tag, body) を順に返す。部分長は配布物の鍵・署名では使われないため拒否する。"""
    offset = 0
    while offset < len(data):
        header = data[offset]
        if not header & 0x80:
            raise SignatureError("OpenPGPのパケットではありません")
        if header & 0x40:
            tag = header & 0x3F
            first = data[offset + 1]
            if first < 192:
                length, offset = first, offset + 2
            elif first < 224:
                length, offset = ((first - 192) << 8) + data[offset + 2] + 192, offset + 3
            elif first == 255:
                length, offset = int.from_bytes(data[offset + 2:offset + 6], "big"), offset + 6
            else:
                raise SignatureError("部分長のパケットには対応しません")
        else:
            tag = (header >> 2) & 0x0F
            kind = header & 0x03
            if kind == 3:
                raise SignatureError("長さ不定のパケットには対応しません")
            size = 1 << kind
            length = int.from_bytes(data[offset + 1:offset + 1 + size], "big")
            offset += 1 + size
        body = data[offset:offset + length]
        if len(body) != length:
            raise SignatureError("パケットが途中で切れています")
        yield tag, body
        offset += length


def _mpi(body: bytes, offset: int) -> tuple[int, int]:
    bits = int.from_bytes(body[offset:offset + 2], "big")
    size = (bits + 7) // 8
    value = body[offset + 2:offset + 2 + size]
    if len(value) != size:
        raise SignatureError("MPIが途中で切れています")
    return int.from_bytes(value, "big"), offset + 2 + size


def public_keys(armored: str) -> list[RsaKey]:
    """主鍵と副鍵のうち、v4のRSA鍵を返す。"""
    keys = []
    for tag, body in packets(dearmor(armored)):
        if tag not in (6, 14) or not body or body[0] != 4:
            continue
        if body[5] not in _RSA_ALGORITHMS:
            continue
        n, offset = _mpi(body, 6)
        e, offset = _mpi(body, offset)
        material = body[:offset]
        fingerprint = hashlib.sha1(b"\x99" + len(material).to_bytes(2, "big") + material).hexdigest().upper()
        keys.append(RsaKey(fingerprint, n, e))
    return keys


def _subpackets(data: bytes):
    offset = 0
    while offset < len(data):
        first = data[offset]
        if first < 192:
            length, offset = first, offset + 1
        elif first < 255:
            length, offset = ((first - 192) << 8) + data[offset + 1] + 192, offset + 2
        else:
            length, offset = int.from_bytes(data[offset + 1:offset + 5], "big"), offset + 5
        if length == 0:
            raise SignatureError("署名の副パケットが不正です")
        yield data[offset] & 0x7F, data[offset + 1:offset + length]
        offset += length


def verify_detached(document: bytes, signature: bytes, keys: list[RsaKey]) -> RsaKey:
    """分離署名を固定した鍵で検証し、署名した鍵を返す。失敗は例外。"""
    try:
        return _verify(document, signature, keys)
    except (IndexError, KeyError) as exc:
        raise SignatureError("署名の形式が不正です") from exc


def _verify(document: bytes, signature: bytes, keys: list[RsaKey]) -> RsaKey:
    if signature[:1] == b"-":
        signature = dearmor(signature.decode("ascii", errors="replace"))
    found = [body for tag, body in packets(signature) if tag == 2]
    if len(found) != 1:
        raise SignatureError("署名パケットが1つではありません")
    body = found[0]
    if body[0] != 4:
        raise SignatureError("v4以外の署名には対応しません")
    kind, algorithm, hash_id = body[1], body[2], body[3]
    if kind != _BINARY_DOCUMENT:
        raise SignatureError("バイナリ文書以外の署名には対応しません")
    if algorithm not in _RSA_ALGORITHMS or hash_id not in _HASHES:
        raise SignatureError("対応していない署名アルゴリズムです")
    hashed_length = int.from_bytes(body[4:6], "big")
    hashed = body[6:6 + hashed_length]
    offset = 6 + hashed_length
    unhashed_length = int.from_bytes(body[offset:offset + 2], "big")
    unhashed = body[offset + 2:offset + 2 + unhashed_length]
    offset += 2 + unhashed_length
    left16 = body[offset:offset + 2]
    value, _ = _mpi(body, offset + 2)

    issuers: set[str] = set()
    for type_, data in [*_subpackets(hashed), *_subpackets(unhashed)]:
        if type_ == 16 and len(data) == 8:
            issuers.add(data.hex().upper())
        elif type_ == 33 and len(data) == 21 and data[0] == 4:
            issuers.add(data[1:].hex().upper())
    candidates = [key for key in keys if key.fingerprint in issuers or key.key_id in issuers]
    if not candidates:
        raise SignatureError("固定した鍵による署名ではありません")

    name, prefix = _HASHES[hash_id]
    digest = hashlib.new(name)
    digest.update(document)
    digest.update(body[:6 + hashed_length])
    digest.update(b"\x04\xff" + (6 + hashed_length).to_bytes(4, "big"))
    expected_hash = digest.digest()
    if expected_hash[:2] != left16:
        raise SignatureError("署名と内容が一致しません")
    for key in candidates:
        size = (key.n.bit_length() + 7) // 8
        if not 0 < value < key.n:
            continue
        encoded = pow(value, key.e, key.n).to_bytes(size, "big")
        digest_info = prefix + expected_hash
        padding = size - len(digest_info) - 3
        if padding >= 8 and encoded == b"\x00\x01" + b"\xff" * padding + b"\x00" + digest_info:
            return key
    raise SignatureError("署名と内容が一致しません")
