"""Fast, deterministic digests and payload sizes for simulated transactions.

The original simulation hashed ``str(sorted(...))`` of transactions that contain
model state tensors. Formatting large tensors to text is both very slow and
lossy (PyTorch truncates large tensor strings), so this module canonicalizes
objects into raw bytes instead. The digest is only used inside this simulation,
so the exact byte layout does not need to match historical runs.
"""

from __future__ import annotations

from hashlib import sha256

import torch


def _canonical_bytes(obj) -> bytes:
    if isinstance(obj, torch.Tensor):
        tensor = obj.detach().to("cpu").contiguous()
        header = f"tensor:{tensor.dtype}:{tuple(tensor.shape)}:".encode("utf-8")
        return header + tensor.numpy().tobytes()
    if isinstance(obj, dict):
        items = sorted(obj.items(), key=lambda item: repr(item[0]))
        return (
            b"dict{"
            + b"".join(
                _canonical_bytes(key) + b"=>" + _canonical_bytes(value)
                for key, value in items
            )
            + b"}"
        )
    if isinstance(obj, (list, tuple)):
        return b"list[" + b"".join(_canonical_bytes(value) for value in obj) + b"]"
    if isinstance(obj, (set, frozenset)):
        values = sorted(obj, key=repr)
        return b"set{" + b"".join(_canonical_bytes(value) for value in values) + b"}"
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return b"scalar:" + repr(obj).encode("utf-8")
    return b"object:" + repr(obj).encode("utf-8")


def stable_digest(obj) -> bytes:
    """Return a deterministic SHA-256 digest of tensors and nested containers."""
    return sha256(_canonical_bytes(obj)).digest()


def payload_size_bytes(obj) -> int:
    """Approximate the serialized payload size of a transaction-like object."""
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return (
            sum(
                len(str(key).encode("utf-8")) + payload_size_bytes(value)
                for key, value in obj.items()
            )
            + 16
        )
    if isinstance(obj, (list, tuple, set, frozenset)):
        return sum(payload_size_bytes(value) for value in obj) + 16
    if isinstance(obj, str):
        return len(obj.encode("utf-8"))
    if isinstance(obj, (int, float, bool)) or obj is None:
        return 8
    return len(repr(obj).encode("utf-8"))
