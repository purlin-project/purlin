import torch

import purlin


class RecordingModule:
    def __init__(self):
        self.calls = []

    def initialize(self, *args):
        self.calls.append(("initialize", args))
        return "ctx"

    def finalize(self, *args):
        self.calls.append(("finalize", args))

    def all_gather(self, *args):
        self.calls.append(("all_gather", args))

    def all_gather_v(self, *args):
        self.calls.append(("all_gather_v", args))

    def all_reduce(self, *args):
        self.calls.append(("all_reduce", args))

    def all_to_all(self, *args):
        self.calls.append(("all_to_all", args))

    def all_to_all_v(self, *args):
        self.calls.append(("all_to_all_v", args))

    def reduce_scatter(self, *args):
        self.calls.append(("reduce_scatter", args))

    def reduce_scatter_v(self, *args):
        self.calls.append(("reduce_scatter_v", args))


def make_handle(mod):
    return purlin.ContextHandle(
        mod,
        "ctx",
        "buf",
        "hdl",
        "sig_buf",
        "sig_hdl",
        "v_sig_buf",
        "v_sig_hdl",
    )


def test_collective_wrappers_dispatch_to_bound_methods():
    mod = RecordingModule()
    handle = make_handle(mod)
    stream_ptr = 123

    src = torch.empty(8, dtype=torch.float32)
    dst = torch.empty(16, dtype=torch.float32)
    byte_splits = [16, 32]

    purlin.all_gather(src, dst, handle, stream_ptr)
    assert mod.calls.pop() == (
        "all_gather",
        (src.data_ptr(), dst.data_ptr(), src.nbytes, "ctx", stream_ptr),
    )

    purlin.all_gather_v(src, dst, byte_splits, handle, stream_ptr)
    assert mod.calls.pop() == (
        "all_gather_v",
        (src.data_ptr(), dst.data_ptr(), byte_splits, "ctx", stream_ptr),
    )

    purlin.all_reduce(src, dst, handle, stream_ptr)
    assert mod.calls.pop() == (
        "all_reduce",
        (
            src.data_ptr(),
            dst.data_ptr(),
            src.nbytes,
            purlin.DataType.FP32,
            "ctx",
            stream_ptr,
            purlin.ReductionMode.NON_DETERMINISTIC,
        ),
    )

    purlin.all_to_all(src, dst, handle, stream_ptr)
    assert mod.calls.pop() == (
        "all_to_all",
        (src.data_ptr(), dst.data_ptr(), src.nbytes, "ctx", stream_ptr),
    )

    purlin.all_to_all_v(src, dst, byte_splits, handle, stream_ptr)
    assert mod.calls.pop() == (
        "all_to_all_v",
        (src.data_ptr(), dst.data_ptr(), byte_splits, "ctx", stream_ptr),
    )

    purlin.reduce_scatter(src, dst, handle, stream_ptr)
    assert mod.calls.pop() == (
        "reduce_scatter",
        (
            src.data_ptr(),
            dst.data_ptr(),
            src.nbytes,
            purlin.DataType.FP32,
            "ctx",
            stream_ptr,
            purlin.ReductionMode.NON_DETERMINISTIC,
        ),
    )

    purlin.reduce_scatter_v(src, dst, byte_splits, handle, stream_ptr)
    assert mod.calls.pop() == (
        "reduce_scatter_v",
        (
            src.data_ptr(),
            dst.data_ptr(),
            byte_splits,
            purlin.DataType.FP32,
            "ctx",
            stream_ptr,
            purlin.ReductionMode.NON_DETERMINISTIC,
        ),
    )


def test_initialize_forwards_symmetric_memory_multicast_pointer(monkeypatch):
    import torch.distributed._symmetric_memory as sym_mem

    from purlin import jit

    class FakeTensor:
        def __init__(self, size, dtype, device):
            self.size = size
            self.dtype = dtype
            self.device = device
            self.zeroed = False

        def zero_(self):
            self.zeroed = True

    class FakeHandle:
        def __init__(self, buffer_ptrs, multicast_ptr=0):
            self.buffer_ptrs = buffer_ptrs
            self.multicast_ptr = multicast_ptr

    mod = RecordingModule()
    tensors = []
    handles = iter(
        [
            FakeHandle([100, 200], multicast_ptr=300),
            FakeHandle([400, 500]),
            FakeHandle([600, 700]),
        ]
    )

    def fake_empty(size, *, dtype, device):
        tensor = FakeTensor(size, dtype, device)
        tensors.append(tensor)
        return tensor

    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    monkeypatch.setattr(sym_mem, "empty", fake_empty)
    monkeypatch.setattr(sym_mem, "rendezvous", lambda tensor, group: next(handles))
    monkeypatch.setattr(jit, "get_compiled", lambda *args, **kwargs: mod)

    device = torch.device("cuda", 0)
    handle = purlin.initialize("group", device, 90, 800)

    assert mod.calls == [
        (
            "initialize",
            (0, 2, [100, 200], 300, [400, 500], [600, 700],
             purlin.STAGING_BUFFER_SIZE, 800),
        )
    ]
    assert all(tensor.zeroed for tensor in tensors)
    assert handle.hdl.multicast_ptr == 300


def test_finalize_releases_python_references_and_is_idempotent():
    mod = RecordingModule()
    handle = make_handle(mod)

    purlin.finalize(handle, 456)

    assert mod.calls == [("finalize", ("ctx", 456))]
    assert handle.ctx is None
    assert handle.buf is None
    assert handle.hdl is None
    assert handle.sig_buf is None
    assert handle.sig_hdl is None
    assert handle.v_sig_buf is None
    assert handle.v_sig_hdl is None

    purlin.finalize(handle, 456)
    assert mod.calls == [("finalize", ("ctx", 456))]


def test_non_contiguous_inputs_are_rejected_before_bound_call():
    mod = RecordingModule()
    handle = make_handle(mod)
    src = torch.empty(4, 4).t()
    dst = torch.empty(4, 4)

    try:
        purlin.all_gather(src, dst, handle, 0)
    except AssertionError:
        pass
    else:
        raise AssertionError("expected non-contiguous input to be rejected")

    assert mod.calls == []


def test_reduction_modes_are_forwarded_and_validated():
    import pytest

    mod = RecordingModule()
    handle = make_handle(mod)
    src = torch.empty(8)
    dst = torch.empty(8)
    for name in ("all_reduce", "reduce_scatter", "reduce_scatter_v"):
        operation = getattr(purlin, name)
        args = (src, dst, [16, 16], handle, 123) if name.endswith("_v") else (src, dst, handle, 123)
        for mode in purlin.ReductionMode:
            operation(*args, reduction_mode=mode)
            assert mod.calls.pop()[1][-1] == int(mode)
        with pytest.raises(ValueError):
            operation(*args, reduction_mode=42)
        assert not mod.calls
