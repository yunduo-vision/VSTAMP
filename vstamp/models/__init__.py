__all__ = ["Fingerprint", "VSTAMP"]


def __getattr__(name: str):
    if name in __all__:
        from .vstamp import Fingerprint, VSTAMP

        return {"Fingerprint": Fingerprint, "VSTAMP": VSTAMP}[name]
    raise AttributeError(name)
