"""Тесты настройки SSL-сертификатов (лечение CERTIFICATE_VERIFY_FAILED)."""

import os

from transcribe_ru.certs import configure_ssl


def test_sets_cert_vars_when_unset():
    env = {}
    configure_ssl(environ=env, ca_bundle="/path/ca.pem")
    assert env["SSL_CERT_FILE"] == "/path/ca.pem"
    assert env["REQUESTS_CA_BUNDLE"] == "/path/ca.pem"


def test_does_not_override_existing():
    env = {"SSL_CERT_FILE": "/мой/custom.pem"}
    configure_ssl(environ=env, ca_bundle="/path/ca.pem")
    assert env["SSL_CERT_FILE"] == "/мой/custom.pem"  # пользовательское не трогаем
    assert env["REQUESTS_CA_BUNDLE"] == "/path/ca.pem"  # незаданное — ставим


def test_uses_certifi_when_no_bundle_given():
    env = {}
    configure_ssl(environ=env)
    import certifi

    assert env["SSL_CERT_FILE"] == certifi.where()
    assert os.path.exists(env["SSL_CERT_FILE"])


def test_ignores_empty_existing_value():
    env = {"SSL_CERT_FILE": ""}
    configure_ssl(environ=env, ca_bundle="/path/ca.pem")
    assert env["SSL_CERT_FILE"] == "/path/ca.pem"


def test_returns_bundle_path():
    assert configure_ssl(environ={}, ca_bundle="/path/ca.pem") == "/path/ca.pem"
