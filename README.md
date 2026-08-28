# Purlin
Purlin is a unified framework providing high-performance, _evolvable_ (1) data movement primitives 
(copy and N-to-1 reduce) which compose through novel mechanisms to (2) state-of-the-art single-kernel collectives 
for the intra-node domain.

Our key innovation is _decoupling orchestration from the datapath_ of collective communication. Purlin separates 
_where and when_ data is moved from _how_ this movement occurs within collectives.  

Every collective is expressed against one fixed orchestration protocol, SNAC, 
which composes with a hardware-aware datapath, the Atom. Collective semantics sit above SNAC, hardware details sits below it, 
and neither leaks into the other.

## 🧨 QuickStart
```bash
uv pip install purlin # best to use a venv here
torchrun --nproc-per-node <num-of-gpus> quickstart.py 
```
