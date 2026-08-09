from __future__ import annotations

import fcntl
import hashlib
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

_PURLIN_VERSION = "v022"
def _verify_dirs() -> None:
    root = Path(__file__).resolve().parent

    if not (root / "CMakeLists.txt").exists():
        raise RuntimeError("JIT CMakeLists.txt not found at package root")


def _load_ext(mod_name: str, so_path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(mod_name, so_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {mod_name} from {so_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cache_key(
    arch: int,
    src: str | dict[str, str],
    mod_name: str,
) -> str:
    if isinstance(src, dict):
        src_key = "\n".join(f"{name}\0{src[name]}" for name in sorted(src))
    else:
        src_key = src

    key_material = (
        f"{mod_name}|arch{arch}|py{sys.version_info[:2]}|{src_key}"
    )
    return hashlib.sha256(key_material.encode()).hexdigest()[:16]


class _BuildLock:
    """Process-safe lock for one JIT cache entry.

    The lock file is intentionally persistent. ``flock`` owns the lock through
    the open file descriptor and releases it automatically if the builder exits,
    so no stale-file recovery or metadata-based ownership check is required.
    """

    def __init__(self, path: Path):
        self.path = path
        self._fd: int | None = None

    def acquire(self, timeout_s: float = 1800.0, poll_s: float = 0.1) -> None:
        if self._fd is not None:
            raise RuntimeError(f"Build lock is already held: {self.path}")

        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o644)
        deadline = time.monotonic() + timeout_s
        try:
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"Timed out waiting for JIT build lock {self.path}."
                        )
                    time.sleep(poll_s)

            metadata = (
                f"host={socket.gethostname()}\n"
                f"pid={os.getpid()}\n"
                f"tid={threading.get_ident()}\n"
                f"time={time.time()}\n"
            ).encode()
            os.ftruncate(fd, 0)
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(fd, metadata)
            os.fsync(fd)
        except BaseException:
            os.close(fd)
            raise

        self._fd = fd

    def release(self) -> None:
        if self._fd is None:
            return

        fd, self._fd = self._fd, None
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def __enter__(self) -> _BuildLock:
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()


def get_compiled(
    arch: int,
    src: str | dict[str, str],
    mod_prefix: str,
    mod_name: str
):
    _verify_dirs()

    cache = Path(
        os.environ.get("PURLIN_CACHE_DIR", str(Path.home() / ".cache" / "purlin_jit" / _PURLIN_VERSION))
    )
    cache.mkdir(parents=True, exist_ok=True)

    key = _cache_key(arch, src, mod_name)

    build_root = cache / f"{key}"
    build_root.mkdir(parents=True, exist_ok=True)

    so_path = build_root / f"{mod_name}.so"
    lock_path = build_root / ".build.lock"

    # Fast path
    if so_path.exists():
        return _load_ext(mod_name, so_path)

    cmake_source_dir = Path(__file__).resolve().parent

    with _BuildLock(lock_path):
        # Another process may have completed the build while this process was
        # waiting for the lock.
        if not so_path.exists():
            host = socket.gethostname()
            pid = os.getpid()
            uniq = f"{host}_pid{pid}"

            gen_dir = build_root / f"gen_{uniq}"
            bdir = build_root / f"build_{uniq}"
            gen_dir.mkdir(exist_ok=True)
            bdir.mkdir(exist_ok=True)

            if isinstance(src, dict):
                generated_sources = []
                for name, content in src.items():
                    generated = gen_dir / name
                    generated.parent.mkdir(parents=True, exist_ok=True)
                    generated.write_text(content)
                    generated_sources.append(generated)
            else:
                generated = gen_dir / f"{mod_prefix}_bindings.cu"
                generated.write_text(src)
                generated_sources = [generated]

            configure_command = [
                "cmake",
                "-S",
                str(cmake_source_dir),
                "-B",
                str(bdir),
                "-G",
                "Ninja",
                f"-DGENERATED_SRC={';'.join(str(path) for path in generated_sources)}",
                f"-DTARGET_MODULE_NAME={mod_name}",
                f"-DCMAKE_CUDA_ARCHITECTURES={arch}",
                f"-DCPM_SOURCE_CACHE={Path.home() / '.cache' / 'cpm'}",
                "-DCMAKE_BUILD_TYPE=Release",
                f"-DARCH={arch}",
            ]

            subprocess.run(
                configure_command,
                check=True,
            )

            subprocess.run(["cmake", "--build", str(bdir), "--parallel"], check=True)

            built = next(bdir.glob(mod_name + "*.so"))

            # Copy into a temp path in build_root, then atomically replace the
            # final path before allowing another process to acquire the lock.
            tmp_so = build_root / f".{mod_name}.{uniq}.tmp.so"
            shutil.copy2(built, tmp_so)
            tmp_so.replace(so_path)
            built.unlink(missing_ok=True)

    return _load_ext(mod_name, so_path)
