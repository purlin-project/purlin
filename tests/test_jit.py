from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Lock

from purlin import jit
from purlin.bindings import purlin_bindings


def test_reduce_scatter_variants_use_separate_translation_units():
    sources = purlin_bindings.substitute(mod_name="purlin_test")

    fixed = sources["purlin_reduce_scatter.cu"]
    variable = sources["purlin_reduce_scatter_v.cu"]

    assert "void reduce_scatter(" in fixed
    assert "void reduce_scatter_v(" not in fixed
    assert "void reduce_scatter_v(" in variable
    assert "void reduce_scatter(" not in variable


def test_context_binding_wires_multicast_staging_aliases():
    sources = purlin_bindings.substitute(mod_name="purlin_test")

    module = sources["purlin_module.cpp"]
    context = sources["purlin_context.cu"]

    assert "const std::uintptr_t& mc_staging_ptr" in module
    assert ".mcStagingTR = mcStagingTR" in context
    assert ".mcStagingLR = mcStagingLR" in context
    assert "mcStagingTR + offsetTR" in context


def test_cache_key_includes_exact_architecture():
    sources = {"module.cpp": "source"}

    key = jit._cache_key(80, sources, "purlin_800")

    assert key != jit._cache_key(86, sources, "purlin_800")
    assert key != jit._cache_key(88, sources, "purlin_800")


def test_cache_key_is_stable_for_source_dictionary_order():
    first = {"b.cu": "b", "a.cu": "a"}
    second = {"a.cu": "a", "b.cu": "b"}

    assert jit._cache_key(80, first, "purlin_800") == jit._cache_key(
        80, second, "purlin_800"
    )


def test_concurrent_callers_only_build_once(tmp_path, monkeypatch):
    monkeypatch.setenv("PURLIN_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(jit, "_verify_dirs", lambda: None)
    monkeypatch.setattr(jit, "_load_ext", lambda mod_name, so_path: (mod_name, so_path))

    mod_name = "purlin_test"
    build_started = Event()
    release_build = Event()
    calls = []
    calls_lock = Lock()

    def fake_run(command, check):
        assert check
        with calls_lock:
            calls.append(command)

        if "--build" in command:
            build_started.set()
            assert release_build.wait(timeout=5)
            build_dir = Path(command[command.index("--build") + 1])
            (build_dir / f"{mod_name}.so").write_bytes(b"extension")

    monkeypatch.setattr(jit.subprocess, "run", fake_run)

    def compile_module():
        return jit.get_compiled(
            arch=80,
            src={"module.cpp": "source"},
            mod_prefix="purlin",
            mod_name=mod_name
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(compile_module)
        assert build_started.wait(timeout=5)
        second = executor.submit(compile_module)
        assert not second.done()
        release_build.set()
        first_result = first.result(timeout=5)
        second_result = second.result(timeout=5)

    configure_calls = [command for command in calls if "-S" in command]
    build_calls = [command for command in calls if "--build" in command]
    assert len(configure_calls) == 1
    assert len(build_calls) == 1
    assert first_result == second_result
    assert first_result[1].read_bytes() == b"extension"
