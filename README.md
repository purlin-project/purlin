# Purlin
Purlin is a unified abstraction for device-initiated, hardware-adaptable, and fusable (1) single-kernel collectives 
and (2) data movement primitives (copy and N-to-1 reduce) for the intra-node domain.

Its key innovation is _decoupling orchestration from the datapath_ of collective communication. Purlin separates 
_where and when_ data is moved from _how_ this movement occurs within collectives.  

Every collective is expressed against one fixed orchestration protocol, SNAC, 
which composes with a hardware-datapath interface, the Atom. Collective semantics sit above SNAC, hardware details sits below it, 
and neither leaks into the other.

## 🧨 QuickStart
```bash
uv pip install purlin # best to use a venv here
torchrun --nproc-per-node <num-of-gpus> quickstart.py 
```
