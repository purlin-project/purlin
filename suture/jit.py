from __future__ import annotations

import threading
from pathlib import Path

def _verify_dirs() -> None:
    from pathlib import Path
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

def get_compiled(arch: int, src: str, mod_prefix: str, mod_name: str):
    import hashlib
    import os
    import shutil
    import socket
    import subprocess
    import sys
    import time

    _verify_dirs()

    cache = Path(os.environ.get("SUTURE_CACHE_DIR", str(Path.home() / ".cache" / "suture_jit")))
    cache.mkdir(parents=True, exist_ok=True)

    key = hashlib.sha256(f"{mod_name}|py{sys.version_info[:2]}|{src}".encode()).hexdigest()[:16]

    build_root = cache / f"{key}"
    build_root.mkdir(parents=True, exist_ok=True)

    so_path = build_root / f"{mod_name}.so"
    lock_path = build_root / ".build.lock"

    # Fast path
    if so_path.exists():
        return _load_ext(mod_name, so_path)

    # Process-unique tag for temp dirs
    host = socket.gethostname()
    pid = os.getpid()
    tid = threading.get_ident()
    uniq = f"{host}_tid{tid}_pid{pid}"

    gen_dir = build_root / f"gen_{uniq}"
    bdir = build_root / f"build_{uniq}"
    gen_dir.mkdir(exist_ok=True)
    bdir.mkdir(exist_ok=True)

    generated = gen_dir / f"{mod_prefix}_bindings.cu"
    generated.write_text(src)

    cmake_source_dir = Path(__file__).resolve().parent

    def _try_acquire_lock() -> bool:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w") as f:
                f.write(f"host={host}\npid={pid}\ntid={tid}\ntime={time.time()}\n")
            return True
        except FileExistsError:
            return False

    def _lock_is_stale() -> bool:
        try:
            fields = dict(
                line.split("=", 1)
                for line in lock_path.read_text().splitlines()
                if "=" in line
            )
        except OSError:
            return False

        lock_host = fields.get("host")
        lock_pid = fields.get("pid")
        if not lock_host or not lock_pid:
            return True
        if lock_host != host:
            return False
        try:
            os.kill(int(lock_pid), 0)
        except ProcessLookupError:
            return True
        except (PermissionError, ValueError):
            return False
        return False

    def _release_lock() -> None:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass

    def _wait_for_artifact(timeout_s: float = 1800.0, poll_s: float = 0.1):
        start = time.time()
        while True:
            if so_path.exists():
                return _load_ext(mod_name, so_path)

            if time.time() - start > timeout_s:
                raise TimeoutError(
                    f"Timed out waiting for JIT artifact {so_path} while another process was building it."
                )

            time.sleep(poll_s)

    package_dir = Path(__file__).resolve().parent
    repo_root = package_dir.parent
    csrc = repo_root / "csrc"
    # Try to become the builder
    have_lock = _try_acquire_lock()

    if not have_lock and _lock_is_stale():
        _release_lock()
        have_lock = _try_acquire_lock()

    if not have_lock:
        # Another process is building. Wait for the final .so to appear.
        return _wait_for_artifact()

    try:
        # Double-check after lock acquisition in case another process finished just before us
        if so_path.exists():
            return _load_ext(mod_name, so_path)

        subprocess.run([
            "cmake", "-S", str(cmake_source_dir), "-B", str(bdir), "-G", "Ninja",
            f"-DSUTURE_SOURCE_DIR={csrc}",
            f"-DGENERATED_SRC={generated}",
            f"-DTARGET_MODULE_NAME={mod_name}",
            f"-DCMAKE_CUDA_ARCHITECTURES={arch}",
            f"-DCPM_SOURCE_CACHE={Path.home() / '.cache' / 'cpm'}",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DARCH={arch}"
        ], check=True)

        subprocess.run([
            "cmake", "--build", str(bdir), "--parallel"
        ], check=True)

        built = next(bdir.glob(mod_name + "*.so"))

        # Copy into a temp path in build_root, then atomically replace final path
        tmp_so = build_root / f".{mod_name}.{uniq}.tmp.so"
        shutil.copy2(built, tmp_so)
        tmp_so.replace(so_path)

    finally:
        _release_lock()

    return _load_ext(mod_name, so_path)
