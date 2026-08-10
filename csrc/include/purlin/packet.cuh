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

    template<typename V>
    __device__ __forceinline__
    void pack(const V& v, const RT& flag_) {
      static_assert(sizeof(V) == sizeof(RT) && alignof(V) == alignof(RT));
      data = cuda::std::bit_cast<RT>(v);
      flag = flag_;
    }
    template<typename V>
    __device__ __forceinline__
    void unpack(V& v) {
      static_assert(sizeof(V) == sizeof(RT) && alignof(V) == alignof(RT));
      v = cuda::std::bit_cast<V>(data);
    }

    __device__ __forceinline__
    void write(const RT value, const RT packetFlag) {
      // The packet contains the complete publication, so compiler ordering of
      // unrelated memory is intentionally not part of this relaxed operation.
      asm volatile(R"ptx({
        .reg .b128 packet;
        mov.b128 packet, {%1, %2};
        st.relaxed.sys.global.b128 [%0], packet;
      })ptx" :: "l"(this), "l"(value), "l"(packetFlag));
    }

    __device__ __forceinline__
    void writeRelease(const RT value, const RT packetFlag) {
      // Use this form when the packet publishes data held elsewhere.
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
      // Paired with writeRelease after relaxed polling observes the flag.
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
}
#endif //PURLIN_REGIME_CUH
