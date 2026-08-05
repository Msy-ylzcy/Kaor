from __future__ import annotations

from threading import RLock


class SecretStore:
    def __init__(self, service_name: str = "Kaor") -> None:
        self.service_name = service_name
        self._memory: dict[str, str] = {}
        self._lock = RLock()

    def get(self, name: str) -> str:
        with self._lock:
            if name in self._memory:
                return self._memory[name]
            try:
                import keyring

                return keyring.get_password(self.service_name, name) or ""
            except Exception:
                return ""

    def set(self, name: str, value: str) -> None:
        with self._lock:
            # Keep explicit writes authoritative for this process. Some keyring
            # backends accept reads but fail writes, so a fallback used only by
            # set() would otherwise be invisible to the next get().
            self._memory[name] = value
            try:
                import keyring

                if value:
                    keyring.set_password(self.service_name, name, value)
                else:
                    try:
                        keyring.delete_password(self.service_name, name)
                    except Exception:
                        pass
                return
            except Exception:
                return
