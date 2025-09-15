import importlib
import pkgutil
from types import ModuleType
from typing import Dict, Any


NODE_CLASS_MAPPINGS: Dict[str, Any] = {}
NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = {}

for finder, name, ispkg in pkgutil.iter_modules(__path__):
    if name.startswith("__"):
        continue
    try:
        mod: ModuleType = importlib.import_module(f"{__name__}.{name}")
    except Exception as e:
        print(f"[was-affine] Warning: failed to import nodes.{name}: {e}")
        continue

    if hasattr(mod, "NODE_CLASS_MAPPINGS"):
        m = getattr(mod, "NODE_CLASS_MAPPINGS")
        if isinstance(m, dict):
            NODE_CLASS_MAPPINGS.update(m)

    if hasattr(mod, "NODE_DISPLAY_NAME_MAPPINGS"):
        m = getattr(mod, "NODE_DISPLAY_NAME_MAPPINGS")
        if isinstance(m, dict):
            NODE_DISPLAY_NAME_MAPPINGS.update(m)

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
