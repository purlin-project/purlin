//
// Created by Osayamen on 4/8/26.
//

#ifndef PURLIN_REGIME_CUH
#define PURLIN_REGIME_CUH
namespace purlin {
  struct alignas(16) LRP {
    using RT = uint64_t;
    RT data;
    RT flag;

    __device__ __forceinline__
    void write(const RT value, const RT packetFlag) {
      asm volatile(R"ptx({
        .reg .b128 packet;
        mov.b128 packet, {%1, %2};
        st.relaxed.sys.global.b128 [%0], packet;
      })ptx" :: "l"(this), "l"(value), "l"(packetFlag));
    }

    __device__ __forceinline__
    void writeRelease(const RT value, const RT packetFlag) {
      asm volatile(R"ptx({
        .reg .b128 packet;
        mov.b128 packet, {%1, %2};
        st.release.sys.global.b128 [%0], packet;
      })ptx" :: "l"(this), "l"(value), "l"(packetFlag) : "memory");
    }

    __device__ __forceinline__
    LRP load() const {
      LRP packet{};
      asm volatile(R"ptx({
        .reg .b128 value;
        ld.relaxed.sys.global.b128 value, [%2];
        mov.b128 {%0, %1}, value;
      })ptx" : "=l"(packet.data), "=l"(packet.flag) : "l"(this));
      return packet;
    }

    __device__ __forceinline__
    LRP loadAcquire() const {
      LRP packet{};
      asm volatile(R"ptx({
        .reg .b128 value;
        ld.acquire.sys.global.b128 value, [%2];
        mov.b128 {%0, %1}, value;
      })ptx" : "=l"(packet.data), "=l"(packet.flag) : "l"(this) : "memory");
      return packet;
    }

    __device__ __forceinline__
    LRP wait(const RT expectedFlag) const {
      LRP packet{};
      do {
        packet = load();
      } while (packet.flag != expectedFlag);
      return packet;
    }

    __device__ __forceinline__
    RT read(const RT expectedFlag) const {
      return wait(expectedFlag).data;
    }

    __device__ __forceinline__
    LRP waitUntilAtLeast(const RT expectedFlag) const {
      LRP packet{};
      do {
        packet = load();
      } while (packet.flag < expectedFlag);
      return packet;
    }
  };

  static_assert(sizeof(LRP) == 16 && alignof(LRP) == 16);

  __device__ __forceinline__
  static void multimemStPacket(void* __restrict__ const& mc, const uint64_t& value,
    const uint64_t& flag) {
    asm volatile(R"ptx({
      .reg .f32 v0, v1, f0, f1;
      mov.b64 {v0, v1}, %1;
      mov.b64 {f0, f1}, %2;
      multimem.st.relaxed.sys.global.v4.f32 [%0], {v0, v1, f0, f1};
    })ptx" :: "l"(mc), "l"(value), "l"(flag) : "memory");
  }
}
#endif //PURLIN_REGIME_CUH
