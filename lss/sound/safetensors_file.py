"""safetensors形式の重みを、追加の依存なしに読む（読み取り専用・memory map）。

形式は「8 byteのheader長（little endian）+ JSON header + 生のtensor bytes」。headerの各tensorは
``dtype``・``shape``・``data_offsets``（data領域の先頭からの [開始, 終了)）を持つ。範囲が重なる・
はみ出す・dtypeと長さが合わないファイルは読まない。
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

_DTYPES = {
    "F32": ("float32", 4), "F16": ("float16", 2), "BF16": ("bfloat16", 2),
    "I64": ("int64", 8), "I32": ("int32", 4), "I16": ("int16", 2), "I8": ("int8", 1),
    "U8": ("uint8", 1), "BOOL": ("bool", 1),
}
_MAX_HEADER_BYTES = 100 * 1024 * 1024


class SafetensorsError(ValueError):
    """safetensorsのheaderか範囲が不正。"""


def read_header(path: Path) -> tuple[dict, int]:
    """(tensor名 -> 記述, data領域の先頭offset) を返す。``__metadata__`` は除く。"""
    path = Path(path)
    size = path.stat().st_size
    with path.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise SafetensorsError(f"safetensorsのheader長を読めません: {path}")
        (length,) = struct.unpack("<Q", raw)
        if length <= 0 or length > _MAX_HEADER_BYTES or 8 + length > size:
            raise SafetensorsError(f"safetensorsのheader長が不正です: {length}")
        header = json.loads(stream.read(length).decode("utf-8"))
    if not isinstance(header, dict):
        raise SafetensorsError("safetensorsのheaderがobjectではありません")
    header.pop("__metadata__", None)
    base = 8 + length
    spans = []
    for name, entry in header.items():
        if not isinstance(entry, dict) or entry.get("dtype") not in _DTYPES:
            raise SafetensorsError(f"未対応のtensor記述です: {name}")
        shape = entry.get("shape")
        offsets = entry.get("data_offsets")
        if (not isinstance(shape, list) or not all(isinstance(n, int) and n >= 0 for n in shape)
                or not isinstance(offsets, list) or len(offsets) != 2
                or not all(isinstance(n, int) for n in offsets)):
            raise SafetensorsError(f"tensorの形か範囲が不正です: {name}")
        count = 1
        for n in shape:
            count *= n
        start, end = offsets
        if start < 0 or end < start or base + end > size or end - start != count * _DTYPES[entry["dtype"]][1]:
            raise SafetensorsError(f"tensorの範囲が不正です: {name}")
        spans.append((start, end, name))
    spans.sort()
    for (_, previous_end, previous), (start, _, name) in zip(spans, spans[1:]):
        if start < previous_end:
            raise SafetensorsError(f"tensorの範囲が重なっています: {previous}, {name}")
    return header, base


def load(path: Path, *, names=None, dtype=None) -> dict:
    """tensorを読み、``{名前: torch.Tensor}`` を返す。``names`` を渡すとその名前だけ読む。

    ``dtype`` を渡すと浮動小数のtensorだけをその型へ変換する（読み込み後のcopy）。
    """
    import numpy as np
    import torch

    path = Path(path)
    header, base = read_header(path)
    wanted = list(header) if names is None else [name for name in names]
    missing = [name for name in wanted if name not in header]
    if missing:
        raise SafetensorsError(f"重みにtensorがありません: {missing[:5]}")
    mapped = np.memmap(path, dtype=np.uint8, mode="r")
    out = {}
    try:
        for name in wanted:
            entry = header[name]
            kind, _ = _DTYPES[entry["dtype"]]
            start, end = entry["data_offsets"]
            raw = torch.frombuffer(bytearray(mapped[base + start:base + end]), dtype=torch.uint8)
            tensor = raw.view(getattr(torch, kind)).reshape(entry["shape"])
            if dtype is not None and tensor.is_floating_point():
                tensor = tensor.to(dtype)
            out[name] = tensor
    finally:
        del mapped
    return out


def copy_into(module, path: Path, *, names=None) -> None:
    """重みを1 tensorずつ ``module`` の同名のparameter・bufferへ写す（全体をCPUのmemoryへ溜めない）。

    ``module`` は読み込む先の装置（GPU）と型で組み終えていること。名前と形は **strict** に照合する:
    ``module`` の名前の集合と読む名前の集合が一致しなければ、何も写さずに例外にする。
    """
    import torch

    path = Path(path)
    header, base = read_header(path)
    wanted = list(header) if names is None else list(names)
    missing = [name for name in wanted if name not in header]
    if missing:
        raise SafetensorsError(f"重みにtensorがありません: {missing[:5]}")
    targets = dict(module.state_dict(keep_vars=True))
    if set(targets) != set(wanted):
        raise SafetensorsError(f"重みとmodelの名前が合いません: missing={sorted(set(targets) - set(wanted))[:5]} "
                               f"unexpected={sorted(set(wanted) - set(targets))[:5]}")
    for name in wanted:
        entry = header[name]
        if list(targets[name].shape) != list(entry["shape"]):
            raise SafetensorsError(f"tensorの形が合いません: {name} {entry['shape']} != {list(targets[name].shape)}")
    # memory mapではなく通常の読み出しで1 tensorずつ読む（読んだ頁がこの処理の常駐memoryに残らない）。
    with path.open("rb") as stream, torch.no_grad():
        for name in sorted(wanted, key=lambda item: header[item]["data_offsets"][0]):
            entry = header[name]
            kind, _ = _DTYPES[entry["dtype"]]
            start, end = entry["data_offsets"]
            buffer = bytearray(end - start)
            stream.seek(base + start)
            if stream.readinto(buffer) != end - start:
                raise SafetensorsError(f"tensorを最後まで読めません: {name}")
            if end > start:
                tensor = torch.frombuffer(buffer, dtype=torch.uint8).view(getattr(torch, kind)).reshape(entry["shape"])
                targets[name].data.copy_(tensor)
                del tensor
            del buffer


def build_empty(factory, *, device: str, prepare=None):
    """``factory()`` のmodelを、parameterだけ中身の無いまま（meta）組み、``device`` へ空の領域として置く。

    parameterは組む間に1つずつmetaへ移すので、CPUのmemoryにはmodel全体を置かない。bufferは従来どおり
    CPUで計算した値のまま ``device`` へ移す（GPUで計算し直すと値が末尾の桁で変わるため）。``prepare`` は
    置く前に型を揃える等の処理（例: 一部をbfloat16へ）。続けて ``copy_into`` で全parameterを埋めること。
    """
    import torch

    original = torch.nn.Module.register_parameter

    def register(module, name, parameter):
        if parameter is not None and parameter.device.type != "meta":
            parameter = torch.nn.Parameter(parameter.detach().to("meta"), requires_grad=parameter.requires_grad)
        original(module, name, parameter)

    torch.nn.Module.register_parameter = register
    try:
        model = factory()
    finally:
        torch.nn.Module.register_parameter = original
    if prepare is not None:
        prepare(model)
    with torch.no_grad():
        for module in model.modules():
            for name, parameter in list(module._parameters.items()):
                if parameter is not None:
                    module._parameters[name] = torch.nn.Parameter(
                        torch.empty(parameter.shape, dtype=parameter.dtype, device=device),
                        requires_grad=parameter.requires_grad)
            for name, buffer in list(module._buffers.items()):
                if buffer is not None:
                    module._buffers[name] = buffer.to(device)
    return model
