"""Движок GigaAM-v3 (`ai-sage/GigaAM-v3`) на PyTorch через transformers.

Conformer/RNN-T-модель от Salute с сильным качеством на русском. Модель грузится
через `trust_remote_code` (свой `modeling_gigaam.py`). Тяжёлые зависимости
(`torch`/`transformers`) импортируются лениво внутри `load`, чтобы импорт самого
модуля движка (и его регистрация в реестре) был дешёвым.
"""

from __future__ import annotations

from transcribe_ru.engines.base import Engine, register

MODEL_ID = "ai-sage/GigaAM-v3"

# Варианты модели (revision на HF). e2e_* дают пунктуацию + нормализацию из коробки.
VARIANTS = ("e2e_rnnt", "e2e_ctc", "rnnt", "ctc")
DEFAULT_VARIANT = "e2e_rnnt"


def _default_model_loader(variant: str, device: str):
    """Загрузить GigaAM-v3 нужного варианта и перенести на устройство."""
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        MODEL_ID, revision=variant, trust_remote_code=True
    )
    return model.to(device)


@register("gigaam")
class GigaAMEngine(Engine):
    name = "gigaam"
    language = "ru"

    def __init__(self, variant: str = DEFAULT_VARIANT, *, model_loader=None):
        if variant not in VARIANTS:
            raise ValueError(
                f"Неизвестный вариант {variant!r}; допустимы: {', '.join(VARIANTS)}"
            )
        self.variant = variant
        self._model_loader = model_loader or _default_model_loader
        self._model = None

    def load(self, device: str) -> None:
        self._model = self._model_loader(self.variant, device)

    def transcribe_segment(self, wav, sr: int) -> str:
        if self._model is None:
            raise RuntimeError("Движок не загружен; сначала вызовите load(device).")
        # Окно <25с (ниже LONGFORM_THRESHOLD), поэтому обычный transcribe,
        # а не transcribe_longform (тот тянет gated pyannote).
        return self._model.transcribe(wav).strip()
