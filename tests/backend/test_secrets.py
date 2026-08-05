from __future__ import annotations

import sys
from types import SimpleNamespace

from backend.secrets import SecretStore


def test_explicit_values_survive_a_keyring_that_reads_but_cannot_write(monkeypatch):
    def fail_write(*_args, **_kwargs):
        raise RuntimeError("credential backend is read-only")

    keyring = SimpleNamespace(
        get_password=lambda *_args, **_kwargs: None,
        set_password=fail_write,
        delete_password=fail_write,
    )
    monkeypatch.setitem(sys.modules, "keyring", keyring)

    store = SecretStore()
    store.set("translation-api-key", "remote-secret")
    assert store.get("translation-api-key") == "remote-secret"

    store.set("translation-api-key", "")
    assert store.get("translation-api-key") == ""
