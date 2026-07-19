from enum import IntEnum
import torch

class ContextHandle:
    __slots__ = ("mod", "ctx", "buf", "hdl", "sig_buf", "sig_hdl", "v_sig_buf", "v_sig_hdl")

    def __init__(self, mod, ctx, sym_buf, sym_hdl, sig_buf, sig_hdl, v_sig_buf, v_sig_hdl):
        self.mod = mod
        self.ctx = ctx
        self.buf = sym_buf
        self.hdl = sym_hdl
        self.sig_buf = sig_buf
        self.sig_hdl = sig_hdl
        self.v_sig_buf = v_sig_buf
        self.v_sig_hdl = v_sig_hdl


STAGING_BUFFER_SIZE = 256 * 1024 * 1024
PACKET_BUFFER_SIZE = 2 * 512 * 1024


class DataType(IntEnum):
    BF16 = 0
    FP16 = 1
    FP8_E4M3 = 2
    FP8_E5M2 = 3
    FP32 = 4


def normalize_arch(arch: int):
    if arch >= 100:
        return 1000
    if arch >= 90:
        return 900
    if arch >= 80:
        return 800
    return 700


def buffer_type(t: torch.dtype):
    if t == torch.float16:
        return DataType.FP16
    if t == torch.bfloat16:
        return DataType.BF16
    if t == torch.float32:
        return DataType.FP32
    if t == torch.float8_e5m2:
        return DataType.FP8_E5M2
    if t == torch.float8_e4m3fn:
        return DataType.FP8_E4M3
    assert False, "invalid type"


def initialize(group, device: torch.device, arch: int, stream_ptr: int):
    import torch.distributed._symmetric_memory as sym_mem
    from . import jit
    from .bindings import purlin_bindings
    rank = torch.distributed.get_rank(group)
    world = torch.distributed.get_world_size(group)
    mod_prefix = "purlin"
    n_arch = normalize_arch(arch)
    mod_name = "purlin_{}".format(n_arch)
    src = purlin_bindings.substitute(mod_name=mod_name)
    mod = jit.get_compiled(arch, src, mod_prefix, mod_name, world)
    # staging buffers
    staging_size = 2 * (STAGING_BUFFER_SIZE + (world * PACKET_BUFFER_SIZE))
    t = sym_mem.empty(staging_size, dtype=torch.uint8, device=device)
    t.zero_()
    # signal pads
    t1 = sym_mem.empty(2 * world, dtype=torch.uint64, device=device)
    t1.zero_()
    # var signal pads
    t2 = sym_mem.empty(8 * world, dtype=torch.uint64, device=device)
    hdl = sym_mem.rendezvous(t, group)
    hdl1 = sym_mem.rendezvous(t1, group)
    hdl2 = sym_mem.rendezvous(t2, group)
    ctx = mod.initialize(rank, world,
                         hdl.buffer_ptrs, hdl1.buffer_ptrs, hdl2.buffer_ptrs, STAGING_BUFFER_SIZE, stream_ptr)
    return ContextHandle(mod, ctx, t, hdl, t1, hdl1, t2, hdl2)


def finalize(handle: ContextHandle, stream_ptr: int):
    if handle.ctx is None:
        return
    handle.mod.finalize(handle.ctx, stream_ptr)
    handle.ctx = None
    handle.hdl = None
    handle.buf = None
    handle.sig_buf = None
    handle.sig_hdl = None
    handle.v_sig_buf = None
    handle.v_sig_hdl = None


def all_gather(in_tensor: torch.Tensor, out_tensor: torch.Tensor, handle: ContextHandle, stream_ptr: int):
    assert in_tensor.is_contiguous()
    assert out_tensor.is_contiguous()
    handle.mod.all_gather(in_tensor.data_ptr(), out_tensor.data_ptr(), in_tensor.numel() * in_tensor.element_size(), handle.ctx, stream_ptr)

def all_gather_v(in_tensor: torch.Tensor, out_tensor: torch.Tensor, bytes_list: list[int], handle: ContextHandle, stream_ptr: int):
    assert in_tensor.is_contiguous()
    assert out_tensor.is_contiguous()
    handle.mod.all_gather_v(in_tensor.data_ptr(), out_tensor.data_ptr(), bytes_list, handle.ctx, stream_ptr)

def all_reduce(in_tensor: torch.Tensor, out_tensor: torch.Tensor, handle: ContextHandle, stream_ptr: int):
    assert in_tensor.is_contiguous()
    assert out_tensor.is_contiguous()
    bt = buffer_type(in_tensor.dtype)
    handle.mod.all_reduce(in_tensor.data_ptr(), out_tensor.data_ptr(), in_tensor.numel() * in_tensor.element_size(), bt, handle.ctx, stream_ptr)


def all_to_all(in_tensor: torch.Tensor, out_tensor: torch.Tensor, handle: ContextHandle, stream_ptr: int):
    assert in_tensor.is_contiguous()
    assert out_tensor.is_contiguous()
    handle.mod.all_to_all(in_tensor.data_ptr(), out_tensor.data_ptr(), in_tensor.numel() * in_tensor.element_size(), handle.ctx, stream_ptr)

def all_to_all_v(in_tensor: torch.Tensor, out_tensor: torch.Tensor, splits: list[int], handle: ContextHandle, stream_ptr: int):
    assert in_tensor.is_contiguous()
    assert out_tensor.is_contiguous()
    handle.mod.all_to_all_v(in_tensor.data_ptr(), out_tensor.data_ptr(), splits, handle.ctx, stream_ptr)


def reduce_scatter(in_tensor: torch.Tensor, out_tensor: torch.Tensor, handle: ContextHandle, stream_ptr: int):
    assert in_tensor.is_contiguous()
    assert out_tensor.is_contiguous()
    bt = buffer_type(in_tensor.dtype)
    handle.mod.reduce_scatter(in_tensor.data_ptr(), out_tensor.data_ptr(), in_tensor.numel() * in_tensor.element_size(), bt, handle.ctx, stream_ptr)


def reduce_scatter_v(in_tensor: torch.Tensor, out_tensor: torch.Tensor, bytes_list: list[int], handle: ContextHandle, stream_ptr: int):
    assert in_tensor.is_contiguous()
    assert out_tensor.is_contiguous()
    bt = buffer_type(in_tensor.dtype)
    handle.mod.reduce_scatter_v(in_tensor.data_ptr(), out_tensor.data_ptr(), bytes_list, bt, handle.ctx, stream_ptr)
