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
        ),
    )


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
