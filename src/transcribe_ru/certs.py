"""Настройка корневых сертификатов для HTTPS.

`gigaam` скачивает модель через `urllib`, который на macOS часто не находит
системные сертификаты → `CERTIFICATE_VERIFY_FAILED`. Указываем Python путь к
бандлу `certifi` через переменные окружения, которые уважают и `ssl`/`urllib`,
и `requests`. Не трогаем значения, заданные пользователем.
"""

from __future__ import annotations

import os

_CERT_VARS = ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE")


def configure_ssl(environ=None, ca_bundle: str | None = None) -> str | None:
    """Прописать путь к корневым сертификатам в окружение, если он не задан.

    Возвращает использованный путь к бандлу (или None, если certifi недоступен
    и путь не передан). Вызывать на старте приложения — до сетевых обращений.
    """
    env = environ if environ is not None else os.environ

    if ca_bundle is None:
        try:
            import certifi

            ca_bundle = certifi.where()
        except Exception:
            return None

    for var in _CERT_VARS:
        if not env.get(var):
            env[var] = ca_bundle
    return ca_bundle
