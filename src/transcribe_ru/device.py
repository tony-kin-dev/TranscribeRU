"""Авто-выбор вычислительного устройства: cuda > mps > cpu."""

from __future__ import annotations

VALID_DEVICES = ("auto", "cuda", "mps", "cpu")


def select_device(requested: str = "auto", *, torch_module=None) -> str:
    """Вернуть устройство для движка.

    `requested="auto"` выбирает лучшее доступное: cuda → mps → cpu.
    Явное значение (`cuda`/`mps`/`cpu`) возвращается как есть.
    `torch_module` инъектируется в тестах; по умолчанию импортируется torch.
    """
    if requested not in VALID_DEVICES:
        raise ValueError(
            f"Неизвестное устройство {requested!r}; допустимы: {', '.join(VALID_DEVICES)}"
        )

    if requested != "auto":
        return requested

    if torch_module is None:
        import torch as torch_module

    if torch_module.cuda.is_available():
        return "cuda"
    if torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"
