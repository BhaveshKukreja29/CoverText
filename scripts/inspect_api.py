import importlib
import inspect
import pkgutil
from pathlib import Path

import covertext

lines = []


def emit(text=""):
    lines.append(text)


for info in pkgutil.walk_packages(covertext.__path__, "covertext."):
    name = info.name
    try:
        mod = importlib.import_module(name)
    except Exception as exc:
        emit(f"\n## {name}  (import failed: {exc})")
        continue
    emit(f"\n## {name}")
    for member_name, obj in inspect.getmembers(mod):
        if member_name.startswith("_") or getattr(obj, "__module__", None) != name:
            continue
        if inspect.isclass(obj):
            emit(f"class {member_name}")
            try:
                emit(f"    __init__{inspect.signature(obj.__init__)}")
            except (ValueError, TypeError):
                pass
            for m_name, m in inspect.getmembers(obj, inspect.isfunction):
                if not m_name.startswith("_"):
                    emit(f"    {m_name}{inspect.signature(m)}")
        elif inspect.isfunction(obj):
            emit(f"def {member_name}{inspect.signature(obj)}")
            doc = (inspect.getdoc(obj) or "").splitlines()
            if doc:
                emit(f"    # {doc[0]}")

out = Path("outputs")
out.mkdir(exist_ok=True)
(out / "api.txt").write_text("\n".join(lines), encoding="utf-8")
print("Saved to outputs/api.txt")