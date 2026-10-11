#pragma once

#include "collectives.h"
#include "nccl_device/utility.h"   // NCCL_HOST_DEVICE_INLINE
#include "bine.h"
#include "device.h"                // ncclPatStep, PatUsed, PatSkipped (+ PatSrcOutput/PatNoLocalCopy)

#ifndef __CUDACC__
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#endif

// ---- geometry: pieces, slots ------------------------------------------------
// A piece   = one block's channel slice, up to `chunk` elements.
// A slot    = one NCCL step, holds P pieces, fully produced and posted before the next.
// Step s moves span=2^s blocks; piece q = e*span + j (e = chunk idx, j = block idx).
struct BineAgGeom {
  int S, P, chunk, nChunks, stride;
  size_t offset, end, count;

  NCCL_HOST_DEVICE_INLINE void init(int S_, size_t count_, size_t channelOffset, size_t channelCount,
                                    size_t chunkCount, size_t slotElems) {
    S = S_; count = count_; offset = channelOffset; end = channelOffset + channelCount;
    chunk   = (int)(chunkCount < slotElems ? chunkCount : slotElems);   // never exceed a slot
    nChunks = chunk > 0 ? (int)((channelCount + chunk - 1) / chunk) : 0;
    stride  = (int)(channelCount < (size_t)chunk ? channelCount : chunk);
    P       = stride > 0 ? (int)(slotElems / stride) : 1;
    if (P < 1) P = 1;
  }
  NCCL_HOST_DEVICE_INLINE int preSlots() const { return nChunks; }
  NCCL_HOST_DEVICE_INLINE int stepSlots(int s) const {
    size_t total = (size_t)nChunks << s;
    return (int)((total + P - 1) / P);
  }
};

// ---- pre-step peers, aliased to partner dims when they coincide -------------
struct BineAgPre { int sendPeer, recvPeer, sendDim, recvDim; };   // peer<0: no pre-step exchange

NCCL_HOST_DEVICE_INLINE BineAgPre bineAgPre(const int* partners, const int* index, const int* order,
                                            int rank, int S) {
  BineAgPre r = {-1, -1, -1, -1};
  const int to = bineSendOrder(order, rank, S);
  if (to == rank) return r;
  const int from = bineSendIndex(index, rank, S);
  r.sendPeer = to;  r.sendDim = S;
  r.recvPeer = from; r.recvDim = S;
  for (int s = 0; s < S; ++s) {
    const int p = bineSendPartner(partners, rank, s, S);
    if (p == to)   r.sendDim = s;
    if (p == from) r.recvDim = s;
  }
  return r;
}

// ---- schedule generator -----------------------------------------------------
// Op order (must equal per-connection order seen by the proxy):
//   pre-step : per chunk, [send][recv]  (or one local copy if redistTo == rank)
//   step s   : per slot g: [send P pieces, post][recv P pieces, post]
// Send and recv alternate per slot so neither side can block on a full FIFO.
class BineAGAlgorithm {
  BineAgGeom g; int rank, myIdx; BineAgPre pre;
  int stage;                 // 0 pre, 1 butterfly, 2 done, 3 empty channel
  int e, half;               // pre-step cursor
  int s, gi, k, span, sendBeg, recvBeg, nSlots; size_t total;   // butterfly cursor

  __device__ __host__ static size_t bmin(size_t a, size_t b) { return a < b ? a : b; }

  __device__ __host__ void enterStep(int step) {
    s = step; span = 1 << s;
    const int blockSize = span << 1;
    const int base = (myIdx / blockSize) * blockSize;
    const bool keepLower = ((myIdx >> s) & 1) == 0;
    sendBeg = keepLower ? base : base + span;
    recvBeg = keepLower ? base + span : base;
    total = (size_t)g.nChunks << s;
    nSlots = g.stepSlots(s);
    gi = 0; k = 0; half = 0;
  }

public:
  __device__ __host__ BineAGAlgorithm(const BineAgGeom& g_, int rank_, int myIdx_, const BineAgPre& pre_)
      : g(g_), rank(rank_), myIdx(myIdx_), pre(pre_), stage(g_.nChunks == 0 ? 3 : 0), e(0), half(0),
        s(0), gi(0), k(0), span(1), sendBeg(0), recvBeg(0), nSlots(0), total(0) {}

  __device__ __host__ int getParallelFactor() const { return 1; }  // single group for now

  __device__ __host__ void getNextOp(struct ncclPatStep* ps) {
    int flags = PatUsed;
    ps->last = 0; ps->nelem = 0; ps->inpIx = ps->outIx = 0;
    ps->sendDim = ps->recvDim = -1; ps->sendOffset = ps->recvOffset = -1;
    ps->stepOffset = 0; ps->postSend = ps->postRecv = 0;

    if (stage == 3) {
      flags |= PatSkipped; ps->last = 2;
    } else if (stage == 0) {
      const size_t eo = g.offset + (size_t)e * g.chunk;
      ps->nelem = (int)bmin(g.chunk, g.end - eo);
      if (pre.sendPeer < 0) {                       // own block -> its position (no-op if in place)
        ps->inpIx = eo; ps->outIx = (size_t)rank * g.count + eo; e++;
      } else if (half == 0) {                       // my block -> redistTo's position `rank`
        ps->sendDim = pre.sendDim; ps->inpIx = eo; ps->sendOffset = 0; ps->postSend = 1;
        flags |= PatNoLocalCopy; half = 1;
      } else {                                      // block myIdx <- redistFrom
        ps->recvDim = pre.recvDim; ps->outIx = (size_t)myIdx * g.count + eo; ps->recvOffset = 0;
        ps->postRecv = 1; half = 0; e++;
      }
      if (e >= g.nChunks) { stage = 1; enterStep(0); }
    } else {
      const int len = (int)bmin(g.P, total - (size_t)gi * g.P);
      const size_t q = (size_t)gi * g.P + k;
      const int j = (int)(q % span);
      const size_t eo = g.offset + (q / span) * g.chunk;
      ps->nelem = (int)bmin(g.chunk, g.end - eo);
      const int off = k * g.stride;
      const int post = (k == len - 1) ? 1 : 0;
      if (half == 0) {                              // send from the output buffer
        ps->sendDim = s; ps->inpIx = ps->outIx = (size_t)(sendBeg + j) * g.count + eo;
        ps->sendOffset = off; ps->postSend = post; flags |= PatSrcOutput;
      } else {                                      // recv into the output buffer
        ps->recvDim = s; ps->outIx = (size_t)(recvBeg + j) * g.count + eo;
        ps->recvOffset = off; ps->postRecv = post;
      }
      if (++k == len) {
        k = 0;
        if (half == 0) half = 1;
        else {
          half = 0;
          if (++gi == nSlots) { if (s + 1 < g.S) enterStep(s + 1); else stage = 2; }
        }
      }
    }
    if (stage == 2) ps->last = 2;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
    cuda::atomic_ref<int, cuda::thread_scope_block> a(ps->flags);
    a.store(flags, cuda::memory_order_release);
#else
    ps->flags = flags;
#endif
  }
};
