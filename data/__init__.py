from .build import build_loader as _build_loader


def build_loader(config, simmim=False):
    if not simmim:
        return _build_loader(config)