/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "device.h"
#include "collectives.h"
#include "primitives.h"
#include "bine.h"

namespace {
  template<typename T, typename RedOp, typename Proto>
  __device__ __forceinline__ void runRing(int tid, int nthreads, struct ncclDevWorkColl* work) {
    ncclRing *ring = &ncclShmem.channel.ring;
    int const *ringRanks = ring->userRanks;
    const int nranks = ncclShmem.comm.nRanks;
    size_t count;
    size_t gridOffset;
    size_t channelCount;
    size_t chunkCount;
    ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T), &count, &gridOffset, &channelCount, &chunkCount);
    size_t offset;
    size_t dataOffset;
    uint32_t nelem;
    int rankDest;

    // Coverity reports that the callee treats &ring->next as an array.  However, due to the use of
    // FanSymmetric<1>, only the first element is ever accessed, so it's fine.
    // coverity[callee_ptr_arith:FALSE]
    Primitives<T, RedOp, FanSymmetric<1>, 0, Proto, 0>
      prims(tid, nthreads, &ring->prev, &ring->next, work->sendbuff, work->recvbuff, work->redOpArg);

    for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
      nelem = min(chunkCount, channelCount - elemOffset);

      dataOffset = gridOffset + elemOffset;
      /////////////// begin ReduceScatter steps ///////////////
      // step 0: push data to next GPU
      rankDest = ringRanks[nranks-1];
      offset = dataOffset + rankDest * count;
      prims.send(offset, nelem);

      // k-2 steps: reduce and copy to next GPU
      for (int j=2; j<nranks; ++j) {
        rankDest = ringRanks[nranks-j];
        offset = dataOffset + rankDest * count;
        prims.recvReduceSend(offset, nelem);
      }

      // step k-1: reduce this buffer and data, which will produce the final result
      rankDest = ringRanks[0];
      offset = dataOffset + rankDest * count;
      prims.recvReduceCopy(offset, dataOffset, nelem, /*postOp=*/true);
    }
  }


// In-place: new[pos] = old[order[pos]], restricted to this channel's column slice.
// Cycle-following, so no scratch buffer is needed.
template <typename T>
__device__ __forceinline__ void bineInPlacePermute(
    int tid, int nthreads, T *buf, const int *orderTbl, int nRanks, int steps,
    ssize_t count, ssize_t gridOffset, ssize_t channelCount)
{
  for (int i = 0; i < nRanks; ++i) {
    // Is i the smallest index of its cycle? (fixed points are skipped)
    int j = bineSendOrder(orderTbl, i, steps);
    if (j == i) continue;
    bool leader = true;
    while (j != i) {
      if (j < i) { leader = false; break; }
      j = bineSendOrder(orderTbl, j, steps);
    }
    if (!leader) continue;

    // Every element column is independent, so each thread walks the cycle alone.
    for (ssize_t e = tid; e < channelCount; e += nthreads) {
      const ssize_t col = gridOffset + e;
      T tmp = buf[(ssize_t)i * count + col];
      int cur = i;
      while (true) {
        const int nxt = bineSendOrder(orderTbl, cur, steps);
        if (nxt == i) { buf[(ssize_t)cur * count + col] = tmp; break; }
        buf[(ssize_t)cur * count + col] = buf[(ssize_t)nxt * count + col];
        cur = nxt;
      }
    }
  }
}

// Bine reduce-scatter kernels, reworked like the allgather ones.
//
// NOT COMPILED OR RUN. Written against the NCCL device headers from memory;
// lines marked VERIFY depend on names in your tree.
//
// The helpers up to the first kernel are identical to bine_allgather_split.cuh,
// except for the 4-buffer bineSendRecvSplit and the added bineForMapped. If
// both headers can land in one TU, move them into a shared header.
//
// ============================== PROXY CONTRACT ==============================
// Assumption (VERIFY in prims_simple.h): every Primitives op, whatever nelem,
// consumes one chunk worth of FIFO steps on its connection. So the proxy only
// needs the NUMBER OF CALLS per (channel, step, direction). Let
//   chunk = chunkCount for that channel, cc = channelCount, nB = blocks in range
//   own   = [ownLo, ownHi) positions of the channel (all positions in slice mode)
//
//   Kernel steps, biggest first (span = nRanks/2 ... 1):
//     Send/Permutation: s = steps-1 ... 0     BlockByBlock/DoubleSend: step 0..steps-1
//   Per step, per channel:
//     send calls = ops(sendRange ∩ own)       recv calls = ops(recvRange ∩ own)
//   Either can be 0 while the other is not (a channel may only send or only
//   receive in a step). Direction counts must be tracked separately.
//
//   ops(range):
//     slice mode, cc <  count : nB * ceil(cc / chunk)         (same as the old per-block count)
//     cc == count (incl. pos mode), identity-mapped (Send, Permutation, DoubleSend):
//                               ceil(nB*count / chunk) per CONTIGUOUS RUN
//                               (DoubleSend: up to 2 runs when the window wraps; sum the runs)
//     BlockByBlock (mapped through order[]): nB * ceil(cc / chunk) always
//
//   Post-step (Send variant only, when myIdx != rank):
//     slice mode: ceil(cc/chunk) send to redistTo and recv from redistFrom, every channel
//     pos mode:   ceil(count/chunk) send on the channel owning position myIdx,
//                 ceil(count/chunk) recv on the channel owning position rank
//
//   pos mode is selected iff allowPos && nCh > 1 && nRanks >= nCh &&
//   count*sizeof(T) < nCh*BINE_POS_SLICE_BYTES. Only count, nCh, nRanks
//   enter, so it is identical on all ranks and channels.
// ============================================================================

// Fuse the final local copy into the last recvReduceCopy (writes straight
// into recvbuff, like the ring does). Set to 0 to fall back to a trailing copy.
#ifndef BINE_RS_FUSE_FINAL
#define BINE_RS_FUSE_FINAL 1
#endif

struct BinePart {
  ssize_t count, gridOffset, channelCount, chunkCount;
  int ownLo, ownHi;
  bool idle;
};

template <typename T, typename Proto>
__device__ __forceinline__ BinePart binePartition(ncclDevWorkColl *work,
                                                  int nRanks,
                                                  bool allowPosPartition) {
  BinePart P;
  ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T), &P.count,
                  &P.gridOffset, &P.channelCount, &P.chunkCount);
  P.ownLo = 0;
  P.ownHi = nRanks;
  P.idle = (P.channelCount == 0) || (P.count == 0);

  // VERIFY: channelLo / channelHi are what ncclCollCbdPart uses internally.
  const int nCh = (int)work->channelHi - (int)work->channelLo + 1;
  const int chIdx = (int)ncclShmem.channelId - (int)work->channelLo;

  const bool posPart =
      allowPosPartition && nCh > 1 && nRanks >= nCh &&
      (size_t)P.count * sizeof(T) < (size_t)nCh * BINE_POS_SLICE_BYTES;

  if (posPart) {
    const int per = (nRanks + nCh - 1) / nCh;
    P.ownLo = min(chIdx * per, nRanks);
    P.ownHi = min(P.ownLo + per, nRanks);
    P.gridOffset = 0;
    P.channelCount = P.count;  // whole blocks
    P.idle = (P.ownLo >= P.ownHi) || (P.count == 0);
  }
  return P;
}

// fn(offset, n) for positions [lo, hi), position == block id. n <= chunkCount.
template <typename Fn>
__device__ __forceinline__ void bineForRange(int lo, int hi, const BinePart &P,
                                             Fn fn) {
  if (hi <= lo) return;
  if (P.channelCount == P.count) {  // whole blocks: one contiguous run
    const ssize_t base = (ssize_t)lo * P.count;
    const ssize_t total = (ssize_t)(hi - lo) * P.count;
    for (ssize_t o = 0; o < total; o += P.chunkCount)
      fn(base + o, (int)min(P.chunkCount, total - o));
  } else {  // slice of every block: strided
    for (ssize_t e = 0; e < P.channelCount; e += P.chunkCount) {
      const int n = (int)min(P.chunkCount, P.channelCount - e);
      for (int j = lo; j < hi; ++j)
        fn((ssize_t)j * P.count + P.gridOffset + e, n);
    }
  }
}

// Same, but position -> block id goes through blkOf(pos). Never contiguous.
template <typename Map, typename Fn>
__device__ __forceinline__ void bineForMapped(int lo, int hi, const BinePart &P,
                                              Map blkOf, Fn fn) {
  if (hi <= lo) return;
  for (ssize_t e = 0; e < P.channelCount; e += P.chunkCount) {
    const int n = (int)min(P.chunkCount, P.channelCount - e);
    for (int j = lo; j < hi; ++j)
      fn((ssize_t)blkOf(j) * P.count + P.gridOffset + e, n);
  }
}

// Circular window [start, start+span) mod nRanks, start < nRanks.
template <typename Fn>
__device__ __forceinline__ void bineForCircRange(int start, int span,
                                                 int nRanks,
                                                 const BinePart &P, Fn fn) {
  const int end = start + span;
  bineForRange(max(start, P.ownLo), min(min(end, nRanks), P.ownHi), P, fn);
  if (end > nRanks) bineForRange(P.ownLo, min(end - nRanks, P.ownHi), P, fn);
}

__device__ __forceinline__ bool bineCircNonEmpty(int start, int span,
                                                 int nRanks,
                                                 const BinePart &P) {
  const int end = start + span;
  return (max(start, P.ownLo) < min(min(end, nRanks), P.ownHi)) ||
         (end > nRanks && P.ownLo < min(end - nRanks, P.ownHi));
}

// Half the threads send, half receive, each with its own Primitives and
// buffer pair (the receive side can write to recvbuff for the fused final
// copy while the send side keeps reading the accumulation buffer).
// A side only constructs Primitives if it has traffic; both ends of a
// connection agree on that, otherwise the Direct ctor handshake hangs.
template <typename T, typename RedOp, typename Proto, typename SendFn,
          typename RecvFn>
__device__ __forceinline__ void bineSendRecvSplit(
    int tid, int nthreads, ncclDevWorkColl *work, const T *sIn, T *sOut,
    const T *rIn, T *rOut, int sendPeer, int recvPeer, bool doSend,
    bool doRecv, SendFn sendFn, RecvFn recvFn) {
  const int nSend = (nthreads / 2 / WARP_SIZE) * WARP_SIZE;
  if (tid < nSend) {
    if (doSend) {
      int peer[1] = {sendPeer};
      Primitives<T, RedOp, FanAsymmetric<0, 1>, /*Direct=*/1, Proto, 0> prims(
          tid, nSend, peer, peer, sIn, sOut, work->redOpArg,
          0 * Proto::MaxGroupWidth, 0, 0, work);
      sendFn(prims);
    }
  } else if (doRecv) {
    int peer[1] = {recvPeer};
    Primitives<T, RedOp, FanAsymmetric<1, 0>, /*Direct=*/1, Proto, 0> prims(
        tid - nSend, nthreads - nSend, peer, peer, rIn, rOut, work->redOpArg,
        1 * Proto::MaxGroupWidth, 0, 0, work);
    recvFn(prims);
  }
}

// ---------------------------------------------------------------------------
// Send variant: halve in position space, then redistribute
// ---------------------------------------------------------------------------
template <typename T, typename RedOp, typename Proto>
__device__ __forceinline__ void runBineSend(int tid, int nthreads,
                                            ncclDevWorkColl *work) {
  ncclBine *bine = &ncclShmem.channel.bine;
  const int steps = bine->nDoublingSteps;
  if (steps == 0 || !bine->index || !bine->order) {
    runRing<T, RedOp, Proto>(tid, nthreads, work);
    return;
  }

  const int nRanks = ncclShmem.comm.nRanks;
  const int rank = ncclShmem.comm.rank;

  const BinePart P = binePartition<T, Proto>(work, nRanks, /*allowPos=*/true);
  if (P.idle) return;
  const ssize_t count = P.count;

  // Reduction happens in place in sendbuff, as before.
  T *accumBuf = (T *)work->sendbuff;
  T *outputBuf = (T *)work->recvbuff;

  const int myIdx = bineSendIndex(bine->index, rank, steps);
  const int redistTo = myIdx;
  const int redistFrom = bineSendOrder(bine->order, rank, steps);
  auto owns = [&](int pos) { return pos >= P.ownLo && pos < P.ownHi; };

  // ---- 1. Halving in position space ----
  for (int s = steps - 1; s >= 0; --s) {
    const int partner = bineSendPartner(bine->partners, rank, s, steps);
    if (partner >= 0) {
      const int span = 1 << s;
      const int base = (myIdx >> (s + 1)) << (s + 1);
      const bool keepLower = ((myIdx >> s) & 1) == 0;
      const int sendBeg = keepLower ? base + span : base;
      const int recvBeg = keepLower ? base : base + span;
      const int sLo = max(sendBeg, P.ownLo), sHi = min(sendBeg + span, P.ownHi);
      const int rLo = max(recvBeg, P.ownLo), rHi = min(recvBeg + span, P.ownHi);
      const bool doPost = (s == 0);

      bineSendRecvSplit<T, RedOp, Proto>(
          tid, nthreads, work, accumBuf, accumBuf, accumBuf, accumBuf, partner,
          partner, sHi > sLo, rHi > rLo,
          [&](auto &prims) {
            bineForRange(sLo, sHi, P, [&](ssize_t off, int n) {
              prims.directSendFromOutput(off, n);
            });
          },
          [&](auto &prims) {
            bineForRange(rLo, rHi, P, [&](ssize_t off, int n) {
              prims.directRecvReduceCopy(off, off, n, doPost);
            });
          });
    }
    __syncthreads();  // step s-1 sends what step s reduced
  }

  // ---- 2. Redistribution: position myIdx holds block `myIdx` ----
  if (redistTo == rank) {
    T *mySeg = accumBuf + (ssize_t)rank * count;
    if (outputBuf != mySeg && owns(rank)) {
      for (ssize_t e = tid; e < P.channelCount; e += nthreads)
        outputBuf[P.gridOffset + e] = mySeg[P.gridOffset + e];
    }
  } else {
    bineSendRecvSplit<T, RedOp, Proto>(
        tid, nthreads, work, accumBuf, outputBuf, accumBuf, outputBuf,
        redistTo, redistFrom, /*doSend=*/owns(myIdx), /*doRecv=*/owns(rank),
        [&](auto &prims) {
          for (ssize_t e = 0; e < P.channelCount; e += P.chunkCount) {
            const int n = (int)min(P.chunkCount, P.channelCount - e);
            prims.directSend((ssize_t)myIdx * count + P.gridOffset + e,
                             P.gridOffset + e, n);
          }
        },
        [&](auto &prims) {
          for (ssize_t e = 0; e < P.channelCount; e += P.chunkCount) {
            const int n = (int)min(P.chunkCount, P.channelCount - e);
            prims.directRecv(P.gridOffset + e, n);
          }
        });
  }
}

// ---------------------------------------------------------------------------
// BlockByBlock: natural layout, positions mapped through order[]
// ---------------------------------------------------------------------------
// Pos mode still pays off here: no contiguity, but each block moves as one
// op of `count` elements on one channel instead of nCh tiny ops.
template <typename T, typename RedOp, typename Proto>
__device__ __forceinline__ void runBineBlockByBlock(int tid, int nthreads,
                                                    ncclDevWorkColl *work) {
  ncclBine *bine = &ncclShmem.channel.bine;
  const int steps = bine->nDoublingSteps;
  if (steps == 0 || !bine->index || !bine->order) {
    runRing<T, RedOp, Proto>(tid, nthreads, work);
    return;
  }

  const int nRanks = ncclShmem.comm.nRanks;
  const int rank = ncclShmem.comm.rank;

  const BinePart P = binePartition<T, Proto>(work, nRanks, /*allowPos=*/true);
  if (P.idle) return;
  const ssize_t count = P.count;

  T *accumBuf = (T *)work->sendbuff;
  T *outputBuf = (T *)work->recvbuff;
  const int myIndex = bineSendIndex(bine->index, rank, steps);
  auto owns = [&](int pos) { return pos >= P.ownLo && pos < P.ownHi; };
  auto blkOf = [&](int pos) { return bineSendOrder(bine->order, pos, steps); };

  const bool fusedFinal =
      BINE_RS_FUSE_FINAL && bine->partners[rank * steps + (steps - 1)] >= 0;

  int low = 0, high = nRanks;
  for (int step = 0; step < steps; ++step) {
    const int mid = (low + high) >> 1;
    const bool keepLower = (myIndex < mid);
    const int span = keepLower ? (high - mid) : (mid - low);
    const int partner = bine->partners[rank * steps + step];

    if (partner >= 0) {
      const int recvBegin = keepLower ? low : mid;
      const int sendBegin = keepLower ? mid : low;
      const int sLo = max(sendBegin, P.ownLo);
      const int sHi = min(sendBegin + span, P.ownHi);
      const int rLo = max(recvBegin, P.ownLo);
      const int rHi = min(recvBegin + span, P.ownHi);
      const bool doPost = (step == steps - 1);
      const bool fuseHere = fusedFinal && doPost;           // span == 1
      const ssize_t finalBlkOff = (ssize_t)blkOf(recvBegin) * count;

      bineSendRecvSplit<T, RedOp, Proto>(
          tid, nthreads, work, accumBuf, accumBuf, accumBuf,
          fuseHere ? outputBuf : accumBuf, partner, partner, sHi > sLo,
          rHi > rLo,
          [&](auto &prims) {
            bineForMapped(sLo, sHi, P, blkOf, [&](ssize_t off, int n) {
              prims.directSendFromOutput(off, n);
            });
          },
          [&](auto &prims) {
            bineForMapped(rLo, rHi, P, blkOf, [&](ssize_t off, int n) {
              prims.directRecvReduceCopy(off, fuseHere ? off - finalBlkOff : off,
                                         n, doPost);
            });
          });
    }
    __syncthreads();

    if (keepLower) high = mid; else low = mid;
  }

  // Position `low` is held by order[low]; should equal `rank`.
  if (!fusedFinal && outputBuf != accumBuf && owns(low)) {
    const ssize_t src = (ssize_t)blkOf(low) * count;
    for (ssize_t e = tid; e < P.channelCount; e += nthreads)
      outputBuf[P.gridOffset + e] = accumBuf[src + P.gridOffset + e];
  }
}

// ---------------------------------------------------------------------------
// DoubleSend: circular windows, natural layout, no permutation
// ---------------------------------------------------------------------------
template <typename T, typename RedOp, typename Proto>
__device__ __forceinline__ void runBineDoubleSend(int tid, int nthreads,
                                                  ncclDevWorkColl *work) {
  ncclBine *bine = &ncclShmem.channel.bine;
  const int steps = bine->nDoublingSteps;
  const int p = ncclShmem.comm.nRanks;
  const int rank = ncclShmem.comm.rank;

  const BinePart P = binePartition<T, Proto>(work, p, /*allowPos=*/true);
  if (P.idle) return;
  const ssize_t count = P.count;

  T *accumBuf = (T *)work->sendbuff;
  T *outputBuf = (T *)work->recvbuff;
  auto owns = [&](int pos) { return pos >= P.ownLo && pos < P.ownHi; };

  // Initial window start (matches the inverted distance-halving Bine scatter).
  int M_down_even = 0, M_up_even = 0;
  for (int g = 0; g < steps; ++g) {
    if (g % 2 == 0) M_down_even += (1 << g);
    else            M_up_even += (1 << g);
  }
  int a = (rank % 2 == 0) ? (rank - M_down_even) % p : (rank - M_up_even) % p;
  if (a < 0) a += p;

  const bool fusedFinal = BINE_RS_FUSE_FINAL && steps > 0 &&
                          bine->dhlvPartners[rank * steps + (steps - 1)] >= 0;

  for (int s = 0; s < steps; ++s) {
    const int partner = bine->dhlvPartners[rank * steps + s];
    const int g = steps - 1 - s;
    const int span = 1 << g;
    const bool sendBottom = ((rank % 2) == (g % 2));
    const int sendStart = (a + (sendBottom ? 0 : span)) % p;
    const int recvStart = (a + (sendBottom ? span : 0)) % p;
    a = recvStart;  // kept half == received half. Updated even if idle this step.

    if (partner >= 0) {
      const bool doPost = (s == steps - 1);
      const bool fuseHere = fusedFinal && doPost;  // span == 1, recvStart == rank

      bineSendRecvSplit<T, RedOp, Proto>(
          tid, nthreads, work, accumBuf, accumBuf, accumBuf,
          fuseHere ? outputBuf : accumBuf, partner, partner,
          bineCircNonEmpty(sendStart, span, p, P),
          bineCircNonEmpty(recvStart, span, p, P),
          [&](auto &prims) {
            bineForCircRange(sendStart, span, p, P, [&](ssize_t off, int n) {
              prims.directSendFromOutput(off, n);
            });
          },
          [&](auto &prims) {
            bineForCircRange(recvStart, span, p, P, [&](ssize_t off, int n) {
              prims.directRecvReduceCopy(
                  off, fuseHere ? off - (ssize_t)rank * count : off, n, doPost);
            });
          });
    }
    __syncthreads();
  }

  // Distance halving leaves block `rank` at position `rank`.
  if (!fusedFinal && outputBuf != accumBuf && owns(rank)) {
    const ssize_t src = (ssize_t)rank * count;
    for (ssize_t e = tid; e < P.channelCount; e += nthreads)
      outputBuf[P.gridOffset + e] = accumBuf[src + P.gridOffset + e];
  }
}

// ---------------------------------------------------------------------------
// Permutation variant (your bineInPlacePermute with the orderTbl signature)
// ---------------------------------------------------------------------------
// Slice mode only: the local permutation moves blocks between positions, which
// crosses channel-owned ranges and would need a cross-channel barrier. Runs are
// contiguous only when channelCount == count (e.g. one channel).
template <typename T, typename RedOp, typename Proto>
__device__ __forceinline__ void runBinePermutation(int tid, int nthreads,
                                                   ncclDevWorkColl *work) {
  ncclBine *bine = &ncclShmem.channel.bine;
  const int steps = bine->nDoublingSteps;
  if (steps == 0 || !bine->index || !bine->order) {
    runRing<T, RedOp, Proto>(tid, nthreads, work);
    return;
  }

  const int nRanks = ncclShmem.comm.nRanks;
  const int rank = ncclShmem.comm.rank;

  const BinePart P = binePartition<T, Proto>(work, nRanks, /*allowPos=*/false);
  if (P.idle) return;
  const ssize_t count = P.count;

  T *accumBuf = (T *)work->sendbuff;
  T *outputBuf = (T *)work->recvbuff;
  const int myIdx = bineSendIndex(bine->index, rank, steps);

  // ---- 1. Local permutation: position p <- block order[p] ----
  // Must finish before any Primitives is built: peers may read my buffer.
  bineInPlacePermute<T>(tid, nthreads, accumBuf, bine->order, nRanks, steps,
                        count, P.gridOffset, P.channelCount);
  __syncthreads();

  const bool fusedFinal =
      BINE_RS_FUSE_FINAL && bineSendPartner(bine->partners, rank, 0, steps) >= 0;

  // ---- 2. Halving in position space ----
  for (int s = steps - 1; s >= 0; --s) {
    const int partner = bineSendPartner(bine->partners, rank, s, steps);
    if (partner >= 0) {
      const int span = 1 << s;
      const int base = (myIdx >> (s + 1)) << (s + 1);
      const bool keepLower = ((myIdx >> s) & 1) == 0;
      const int sendBeg = keepLower ? base + span : base;
      const int recvBeg = keepLower ? base : base + span;
      const bool doPost = (s == 0);
      const bool fuseHere = fusedFinal && doPost;  // span == 1, recvBeg == myIdx

      bineSendRecvSplit<T, RedOp, Proto>(
          tid, nthreads, work, accumBuf, accumBuf, accumBuf,
          fuseHere ? outputBuf : accumBuf, partner, partner, true, true,
          [&](auto &prims) {
            bineForRange(sendBeg, sendBeg + span, P,
                         [&](ssize_t off, int n) {
                           prims.directSendFromOutput(off, n);
                         });
          },
          [&](auto &prims) {
            bineForRange(recvBeg, recvBeg + span, P, [&](ssize_t off, int n) {
              prims.directRecvReduceCopy(
                  off, fuseHere ? off - (ssize_t)myIdx * count : off, n, doPost);
            });
          });
    }
    __syncthreads();
  }

  // ---- 3. Final copy: position myIdx holds block `rank` ----
  T *mySeg = accumBuf + (ssize_t)myIdx * count;
  if (!fusedFinal && outputBuf != mySeg) {
    for (ssize_t e = tid; e < P.channelCount; e += nthreads)
      outputBuf[P.gridOffset + e] = mySeg[P.gridOffset + e];
  }
}

template <typename T, typename RedOp, typename Proto>
__device__ __forceinline__ void runBineTree(int tid, int nthreads, ncclDevWorkColl* work) {
  ncclBine* bine = &ncclShmem.channel.bine;
  const int nSteps = bine->nSteps;
  const int nRanks = ncclShmem.comm.nRanks;
  const int rank   = ncclShmem.comm.rank;
 
  // blockStride = elements per rank (stride between blocks in sendbuff), from `count`.
  ssize_t blockStride, gridOffset, channelCount, chunkCount;
  ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T), &blockStride,
                  &gridOffset, &channelCount, &chunkCount);
  if (channelCount == 0) return;   // proxy registers nothing for empty channels
 
  // Degenerate: single rank.
  if (nRanks == 1) {
    const T* in = (const T*)work->sendbuff;
    T* out = (T*)work->recvbuff;
    if (in != out)
      for (ssize_t e = tid; e < channelCount; e += nthreads)
        out[gridOffset + e] = in[gridOffset + e];
    return;
  }
 
  const int root = ncclBineArRoot(ncclShmem.channelId, nRanks);
  int children[BINE_AR_MAX_FAN];     // ascending step order, -1 padded
  int nChildren;
  ncclBineTreeChildrenOf(bine->send, nRanks, nSteps, root, rank, BINE_AR_MAX_FAN, children, &nChildren);
  const int parent = ncclBineTreeParentOf(bine->recv, nRanks, nSteps, root, rank);
 
  // ---------------- root: fused reduce + scatter ----------------
  if (parent < 0) {
    for (ssize_t e = 0; e < channelCount; e += chunkCount) {
      const ssize_t off = gridOffset + e;
      const int nelem = (int)min(chunkCount, channelCount - e);
 
      // Position 0 of the preorder: the root's own block.
      {
        Primitives<T, RedOp, FanAsymmetric<BINE_AR_MAX_FAN, 0>, /*Direct=*/0, Proto, 0>
            prims(tid, nthreads, children, nullptr, work->sendbuff, work->recvbuff,
                  work->redOpArg, 0, 0, 0, work);
        prims.recvReduceCopy((ssize_t)rank * blockStride + off, off, nelem, /*postOp=*/true);
      }
      // Then preorder(child) for each root child, reduced from ALL children, sent to that child.
      for (int i = 0; i < nChildren; ++i) {
        Primitives<T, RedOp, FanAsymmetric<BINE_AR_MAX_FAN, 1>, /*Direct=*/0, Proto, 0>
            prims(tid, nthreads, children, &children[i], work->sendbuff, work->recvbuff,
                  work->redOpArg, 0, 0, 0, work);
        ncclBinePreorder it;
        it.init(bine->send, bine->recv, nRanks, nSteps, root, children[i]);
        int b;
        while (it.next(&b))
          prims.recvReduceSend((ssize_t)b * blockStride + off, nelem, /*postOp=*/true);
      }
    }
    return;
  }
 
  // ---------------- non-root: split threads (same split as runBineTree) ----------------
  int nthreadsSplit;
  if (Proto::Id == NCCL_PROTO_SIMPLE) {
    nthreadsSplit = nthreads / 2;
    if (nthreadsSplit >= 256) nthreadsSplit += 64;
  } else {
    nthreadsSplit = (nthreads * 7 / (10 * WARP_SIZE)) * WARP_SIZE;
  }
 
  if (tid < nthreadsSplit) {
    // ---- REDUCE up: all nRanks blocks, global preorder ----
    Primitives<T, RedOp, FanAsymmetric<BINE_AR_MAX_FAN, 1>, /*Direct=*/0, Proto, 0>
        prims(tid, nthreadsSplit, children, &parent, work->sendbuff, work->recvbuff,
              work->redOpArg, 0 * Proto::MaxGroupWidth, 0, 0, work);
    for (ssize_t e = 0; e < channelCount; e += chunkCount) {
      const ssize_t off = gridOffset + e;
      const int nelem = (int)min(chunkCount, channelCount - e);
      ncclBinePreorder it;
      it.init(bine->send, bine->recv, nRanks, nSteps, root, root);
      int b;
      while (it.next(&b)) {
        const ssize_t in = (ssize_t)b * blockStride + off;
        if (nChildren == 0) prims.send(in, nelem);
        else                prims.recvReduceSend(in, nelem);
      }
    }
  } else {
    // ---- SCATTER down: preorder(rank) = own block, then each child's preorder ----
    const int t = tid - nthreadsSplit;
    const int nt = nthreads - nthreadsSplit;
    int cCount[BINE_AR_MAX_FAN];
    for (int i = 0; i < BINE_AR_MAX_FAN; ++i)
      cCount[i] = (i < nChildren) ? ncclBineTreeSubtreeSize(bine->recv, nRanks, nSteps, root, children[i]) : 0;
 
    for (ssize_t e = 0; e < channelCount; e += chunkCount) {
      const ssize_t off = gridOffset + e;
      const int nelem = (int)min(chunkCount, channelCount - e);
      {
        Primitives<T, RedOp, FanAsymmetric<1, 0>, /*Direct=*/0, Proto, 0>
            prims(t, nt, &parent, nullptr, work->sendbuff, work->recvbuff,
                  work->redOpArg, 1 * Proto::MaxGroupWidth, 0, 0, work);
        prims.recv(off, nelem);                       // my reduced block -> recvbuff
      }
      for (int i = 0; i < nChildren; ++i) {
        Primitives<T, RedOp, FanAsymmetric<1, 1>, /*Direct=*/0, Proto, 0>
            prims(t, nt, &parent, &children[i], work->sendbuff, work->recvbuff,
                  work->redOpArg, 1 * Proto::MaxGroupWidth, 0, 0, work);
        for (int k = 0; k < cCount[i]; ++k) prims.recvSend(nelem);   // pure relay
      }
    }
  }
}
}

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncReduceScatter, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    using Proto = ProtoSimple<REDUCESCATTER_CHUNKSTEPS/REDUCESCATTER_SLICESTEPS, REDUCESCATTER_SLICESTEPS>;
    runRing<T, RedOp, Proto>(tid, nthreads, work);
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncReduceScatter, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    runRing<T, RedOp, ProtoLL>(tid, nthreads, work);
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncReduceScatter, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL128> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    runRing<T, RedOp, ProtoLL128>(tid, nthreads, work);
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncReduceScatter, T, RedOp, NCCL_ALGO_PAT, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
#if __CUDA_ARCH__ >= 600
    using Proto = ProtoSimple<1, 1>;
    const int nranks = ncclShmem.comm.nRanks;
    const int rank = ncclShmem.comm.rank;
    size_t count, channelOffset, channelCount, chunkCount;
    ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T), &count, &channelOffset, &channelCount, &chunkCount);

    static constexpr int nworkers = NCCL_PAT_NWORKERS;
    struct ncclPatShmem* shmem = (struct ncclPatShmem*)ncclScratchForWarp(0);
    uint64_t pollCount = 0;
    __syncthreads(); // Don't start using shared mem until everyone arrives
    for (int i=tid; i<NCCL_SHMEM_PAT_STEPS; i+=nthreads) shmem->patSteps[i].flags = 0;
    if (tid == 0) shmem->localAccSize = 0;
    if (tid == nworkers) shmem->parallelFactor = 0;
    __syncthreads();

    if (tid == nworkers) { // Algo computation thread
      PatRSAlgorithm<T> patAlgo(chunkCount*sizeof(T), NCCL_STEPS, NCCL_PAT_NWORKERS/WARP_SIZE, channelOffset, channelOffset + channelCount, count, chunkCount, rank, nranks);
      int parallelFactor = shmem->parallelFactor = patAlgo.getParallelFactor();
      int step = 0;
      while (1) {
        struct ncclPatStep* ps = shmem->patSteps+(step%NCCL_SHMEM_PAT_STEPS);
        cuda::atomic_ref<int, cuda::thread_scope_block> poll(ps->flags);
        while (poll.load(cuda::memory_order_acquire) != 0) pollCount++; // Wait for workers to be done with step 'step-NCCL_SHMEM_PAT_STEPS'
        patAlgo.getNextOp(ps);
        int last = ps->last;
        step++;
        if (last == 2) break;
      }
    } else if (tid < nworkers) { // Worker threads
      T *inputBuf = (T*)work->sendbuff;
      T *outputBuf = (T*)work->recvbuff;
      int parallelFactor = 0;
      volatile int* pfPtr = &shmem->parallelFactor;
      while (parallelFactor == 0) parallelFactor = *pfPtr;

      int groupSize = nworkers/(WARP_SIZE*parallelFactor) * WARP_SIZE;
      int group = tid / groupSize;
      int nGroups = nworkers / groupSize;
      int tidInGroup = tid - group*groupSize;
      // We don't use recvPeers/sendPeers so let's pass shmem structs instead
      Primitives<T, RedOp, FanSymmetric<1>, 0, Proto, 0> prims
        (tidInGroup, groupSize, (int*)shmem->recvDims, (int*)shmem->sendDims, inputBuf, outputBuf, work->redOpArg, group, 0, 0, nullptr, nullptr, 0, primsModePatRs);

      int step = group;
      while(1) {
        struct ncclPatStep* ps = shmem->patSteps+(step%NCCL_SHMEM_PAT_STEPS);
        cuda::atomic_ref<int, cuda::thread_scope_block> poll(ps->flags);
        while (poll.load(cuda::memory_order_acquire) == 0) pollCount++; // Wait for compute thread
        int last = ps->last;
        prims.patReduce(ps, shmem);
        if (tidInGroup == 0) poll.store(0, cuda::memory_order_release); // Return element to compute thread
        if (last) break;
        step += nGroups;
      }
    }
#endif
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncReduceScatter, T, RedOp, NCCL_ALGO_NVLS, NCCL_PROTO_SIMPLE> {
  template<bool ReduceSendNotRecv>
  struct Scatterer {
    struct ncclDevWorkColl* work;
    int chunkCount;
    ssize_t railGridOffset;

    template<int SlicePerChunk, int MinSrcs, int MaxSrcs, int MinDsts, int MaxDsts, int MultimemSrcs, int MultimemDsts>
    __device__ __forceinline__ void operator()(
        int tid, int tn, int slice, int maxSliceSize,
        int nSrcs, void** srcPtrs, int nDsts, void** dstPtrs, int32_t* dstSizes, uint32_t sendDirectFlag, uint32_t recvDirectFlag
      ) {
      static_assert(SlicePerChunk == 1, "require: SlicePerChunk==1");
      static_assert(MaxDsts <= 1 || MaxSrcs <= 1, "require: MaxDsts<=1 || MaxSrcs<=1");

      struct ncclNvls* nvls = &ncclShmem.channel.nvls;
      int nNodes = ncclShmem.comm.nNodes;
      int nRails = nvls->nHeads;
      int part = ncclShmem.channelId - work->channelLo;
      void* inbuf = (void*)work->sendbuff;
      ssize_t countPerRank = work->collnet.count;

      ssize_t railAllBeg = min(railGridOffset + part * chunkCount, nNodes * countPerRank);
      ssize_t railAllEnd = min(railAllBeg + chunkCount, nNodes * countPerRank);
      int railAllSize = railAllEnd - railAllBeg;
      int rail = nvls->headRank;
      int dst = 0;
      if (ReduceSendNotRecv) {
        if (work->regUsed) return;
        rail = 0;
        nSrcs = 1;
      } else {
        rail = nvls->headRank;
      }
      if (tid < nDsts) dstSizes[tid] = railAllSize;
      do {
        int node = railAllBeg / countPerRank;
        int railAllOffset = 0;
        while (railAllOffset < railAllSize) {
          ssize_t railOneBeg = node * countPerRank;
          ssize_t railOneEnd = railOneBeg + countPerRank;
          ssize_t railOneOffset = (railAllBeg + railAllOffset) - railOneBeg;
          int delta = min(railAllEnd, railOneEnd) - (railAllBeg + railAllOffset);
          int rank = ncclShmem.comm.collNetDenseToUserRank[node * nRails + rail];
          ssize_t userOneBeg = rank * countPerRank + railOneOffset;
          if (nDsts != 0) {
            reduceCopy<ncclCollUnroll(), RedOp, T,
              /*MultimemSrcs=*/MultimemSrcs, 1, 1 + MaxSrcs,
              /*MultimemDsts,MinDsts,MaxDsts=*/MultimemDsts, 1, 1,
              /*PreOpSrcs=*/1>
              (tid, tn, work->redOpArg, false,
                /*nSrcs=*/nSrcs, [=]__device__(int s) {
              return work->regUsed ? (T*)srcPtrs[s] + userOneBeg :
                !ReduceSendNotRecv ? (T*)srcPtrs[s] + railAllOffset:
                (T*)inbuf + userOneBeg;
            },
                /*nDsts=*/1, [=]__device__(int d/*==0*/) {
              return (T*)dstPtrs[dst] + railAllOffset;
            }, delta);
          }
          railAllOffset += delta;
          node += 1;
        }
        dst += 1;
        rail += 1;
      } while (ReduceSendNotRecv && dst < nRails);
    }
  };

  __device__ __forceinline__ void run(int tid, int/*nthreads*/, struct ncclDevWorkColl* work) {
    struct ncclNvls* nvls = &ncclShmem.channel.nvls;
    int nelem;

    /* if we are direct NVLS, we only need to allocate 1 warp to scatter for sync;
     * if not, based on #ranks, we allocate 7 or 5 warps to reduce to saturate bandwidth
     * and the rest are allocated to scatter. */
    const int nThreadsNetRecv = work->oneNode ? 0 : (work->netRegUsed ? WARP_SIZE :  6 * WARP_SIZE);
    const int nThreadsScatter = work->regUsed ? roundUp(nvls->nHeads << 2, WARP_SIZE) : 8 * WARP_SIZE;
    const int nThreadsReduce = NCCL_MAX_NTHREADS - nThreadsNetRecv - nThreadsScatter;
    const int tidEndNetRecv = nThreadsNetRecv;
    const int tidEndScatter = tidEndNetRecv + nThreadsScatter;
    const int tidEndReduce = tidEndScatter + nThreadsReduce;

    if (work->oneNode) {
      const int rank = ncclShmem.comm.rank;
      size_t offset;
      size_t count, gridOffset, channelCount, chunkCount;
      ncclCollCbdPart(work, ncclShmem.channelId, NCCL_PROTO_SIMPLE, sizeof(T), &count, &gridOffset, &channelCount, &chunkCount);
      if (!work->regUsed) {
        if (tid < tidEndScatter) {
          // Scatter
          using Proto = ProtoSimple<1, 1, COLL_UNROLL>;
          Primitives<T, RedOp, FanAsymmetric<0, NCCL_MAX_NVLS_ARITY>, /*Direct=*/0, Proto, 0>
            prims(tid, nThreadsScatter, NULL, nvls->up, work->sendbuff, NULL,
              work->redOpArg, 0 * Proto::MaxGroupWidth, 1, 1);
          for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
            offset = gridOffset + elemOffset;
            nelem = min(chunkCount, channelCount - elemOffset);
            prims.scatter(offset, nvls->nHeads * count, nelem, count, -1, 0);
          }
          // coverity[overrun-call] => Coverity think prims.index can be greater than 1
        } else if (tid < tidEndReduce) {
          // Reduce through NVLS
          using Proto = ProtoSimple<1, 1, COLL_UNROLL, 1, 0>;
          Primitives<T, RedOp, FanAsymmetric<1, 0>, /*Direct=*/0, Proto, 0>
            prims(tid - tidEndScatter, nThreadsReduce, &nvls->down, NULL, NULL, work->recvbuff,
              work->redOpArg, 3 * Proto::MaxGroupWidth, 0, 0);
          for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
            offset = gridOffset + elemOffset;
            nelem = min(chunkCount, channelCount - elemOffset);
            prims.recv(offset, nelem);
          }
        }
      } else {
        if (tid < tidEndScatter) {
          // Scatter
          using Proto = ProtoSimple<1, 1, COLL_UNROLL>;
          Primitives<T, RedOp, FanSymmetric<NCCL_MAX_NVLS_ARITY>, /*Direct=*/0, Proto, 0>
            prims(tid, nThreadsScatter, nvls->up, nvls->up, NULL, NULL,
              work->redOpArg, 0 * Proto::MaxGroupWidth, 1, 1);
          for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
            prims.scatter(0, 0, 0, 0, -1, 0);
          }

          /* gather used as sync */
          prims.gather(0, 0, 0, 0, -1, 0);
        } else if (tid < tidEndReduce) {
          // Reduce through NVLS
          using Proto = ProtoSimple<1, 1, COLL_UNROLL, 1, 0>;
          Primitives<T, RedOp, FanSymmetric<1>, /*Direct=*/1, Proto, 0>
            prims(tid - tidEndScatter, nThreadsReduce, &nvls->down, &nvls->down, NULL, work->recvbuff,
              work->redOpArg, 3 * Proto::MaxGroupWidth, 0, 0, work);
          for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
            size_t outOffset = gridOffset + elemOffset;
            size_t inpOffset = outOffset + rank * count;
            nelem = min(chunkCount, channelCount - elemOffset);
            // Coverity complains about a possible overrun inside the method invoked below, but that's actually
            // a false positive.
            // coverity[overrun-call:FALSE]
            prims.directRecvCopy(inpOffset, outOffset, nelem);
          }

          /* send for sync */
          prims.send(0, 0);
        }
      }
    } else {
      // multi-node
      int nNodes = ncclShmem.comm.nNodes;
      int part = ncclShmem.channelId - work->channelLo;
      ssize_t countPerRank = work->collnet.count;
      const int nChannels = work->channelHi - work->channelLo + 1;
      ssize_t chunkCount = work->collnet.chunkCount;
      if (tid < tidEndNetRecv) {
        using Proto = ProtoSimple<1, 1, COLL_UNROLL>;
        if (work->netRegUsed) {
          if (tid == 0) {
            int steps = (int)divUp(nNodes * countPerRank, nChannels * chunkCount);
            Primitives<T, RedOp, FanAsymmetric<1, 0>, /*Direct=*/0, Proto, 0>::recvPeerNotify(nvls->out, 0, steps);
          }
          __syncwarp();
        } else {
          Primitives<T, RedOp, FanAsymmetric<1, 0>, /*Direct=*/0, Proto, 0>
            prims(tid, nThreadsNetRecv, &nvls->out, nullptr, nullptr, work->recvbuff,
              work->redOpArg, 0 * Proto::MaxGroupWidth, 0, 0);
          for (ssize_t railGridOffset = 0; railGridOffset < nNodes * countPerRank; railGridOffset += nChannels * chunkCount) {
            ssize_t railAllBeg = railGridOffset + part * chunkCount;
            ssize_t railAllEnd = min(railAllBeg + chunkCount, nNodes * countPerRank);
            ssize_t railOneBeg = ncclShmem.comm.node * countPerRank;
            ssize_t railOneEnd = railOneBeg + countPerRank;
            ssize_t beg = max(railAllBeg, railOneBeg);
            ssize_t end = min(railAllEnd, railOneEnd);
            prims.recv(beg - railOneBeg, max(ssize_t(0), end - beg), /*postOp=*/true);
          }
        }
      } else {
        if (tid < tidEndScatter) {
          using Proto = ProtoSimple<1, 1, COLL_UNROLL>;
          Primitives<T, RedOp, FanAsymmetric<0, NCCL_MAX_NVLS_ARITY>, /*Direct=*/1, Proto, 0>
            prims(tid - tidEndNetRecv, nThreadsScatter, nullptr, nvls->up, work->sendbuff, nullptr,
              work->redOpArg, 1 * Proto::MaxGroupWidth, 1, 1, work);
          for (ssize_t railGridOffset = 0; railGridOffset < nNodes * countPerRank; railGridOffset += nChannels * chunkCount) {
            Scatterer</*ReduceSendNotRecv=*/true> scat;
            scat.work = work;
            scat.chunkCount = chunkCount;
            scat.railGridOffset = railGridOffset;
            prims.template process</*Recv=*/0, /*Send=*/1>(scat);
          }
        } else if (tid < tidEndReduce) {
          using Proto = ProtoSimple<1, 1, COLL_UNROLL, 1, 0>;
          Primitives<T, RedOp, FanSymmetric<1>, /*Direct=*/1, Proto, 0>
            prims(tid - tidEndScatter, nThreadsReduce, &nvls->down, &nvls->out, nullptr, nullptr,
              work->redOpArg, 2 * Proto::MaxGroupWidth, 0, 1, work);
          for (ssize_t railGridOffset = 0; railGridOffset < nNodes * countPerRank; railGridOffset += nChannels * chunkCount) {
            Scatterer</*ReduceSendNotRecv=*/false> scat;
            scat.work = work;
            scat.chunkCount = chunkCount;
            scat.railGridOffset = railGridOffset;
            prims.template process</*Recv=*/1, /*Send=*/1>(scat);
          }
        }
      }
    }
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncReduceScatter, T, RedOp, NCCL_ALGO_COLLNET_DIRECT, NCCL_PROTO_SIMPLE> {
  template<bool ReduceSendNotRecv>
  struct Scatterer {
    struct ncclDevWorkColl* work;
    int chunkSize;
    ssize_t railGridOffset;

    template<int SlicePerChunk, int MinSrcs, int MaxSrcs, int MinDsts, int MaxDsts, int MultimemSrcs, int MultimemDsts>
    __device__ __forceinline__ void operator()(
        int tid, int tn, int slice, int maxSliceSize,
        int nSrcs, void** srcPtrs, int nDsts, void** dstPtrs, int32_t* dstSizes, uint32_t sendDirectFlag, uint32_t recvDirectFlag
      ) {
      static_assert(SlicePerChunk==1, "require: SlicePerChunk==1");
      static_assert(MaxDsts<=1 || MaxSrcs<=1, "require: MaxDsts<=1 || MaxSrcs<=1");

      struct ncclDirect* direct = &ncclShmem.channel.collnetDirect;
      int nNodes = ncclShmem.comm.nNodes;
      int nRails = direct->nHeads;
      int part = ncclShmem.channelId - work->channelLo;
      void* inbuf = (void*)work->sendbuff;
      ssize_t countPerRank = work->collnet.count;

      ssize_t railAllBeg = min(railGridOffset + part*chunkSize, nNodes*countPerRank);
      ssize_t railAllEnd = min(railAllBeg + chunkSize, nNodes*countPerRank);
      int railAllSize = railAllEnd - railAllBeg;
      if (tid < nDsts) dstSizes[tid] = railAllSize;

      int dst = 0;
      int rail;
      if (!ReduceSendNotRecv) {
        rail = direct->headRank;
      } else {
        rail = direct->headRank+1;
        if (rail == nRails) rail = 0;
      }
      do {
        int node = railAllBeg/countPerRank;
        int railAllOffset = 0;
        while (railAllOffset < railAllSize) {
          ssize_t railOneBeg = node*countPerRank;
          ssize_t railOneEnd = railOneBeg + countPerRank;
          ssize_t railOneOffset = (railAllBeg+railAllOffset) - railOneBeg;
          int delta = min(railAllEnd, railOneEnd) - (railAllBeg+railAllOffset);
          int rank = ncclShmem.comm.collNetDenseToUserRank[node*nRails + rail];
          ssize_t userOneBeg = rank*countPerRank + railOneOffset;
          if (nDsts != 0) {
            reduceCopy<ncclCollUnroll(), RedOp, T,
                     /*MultimemSrcs=*/0, 1+MinSrcs, 1+MaxSrcs,
                     /*MultimemDsts,MinDsts,MaxDsts=*/0,1,1,
                     /*PreOpSrcs=*/1>
            (tid, tn, work->redOpArg, false,
             /*nSrcs=*/1+nSrcs, [=]__device__(int s) {
               return s==0 ? (T*)inbuf + userOneBeg
                           : work->regUsed && (recvDirectFlag & NCCL_P2P_READ)
                           ? (T*)srcPtrs[s-1] + userOneBeg
                           : (T*)srcPtrs[s-1] + railAllOffset;
             },
             /*nDsts=*/1, [=]__device__(int d/*==0*/) {
               return (T*)dstPtrs[dst] + railAllOffset;
             },
             delta);
          }
          railAllOffset += delta;
          node += 1;
        }
        dst += 1;
        rail += 1;
        if (rail == nRails) rail = 0;
      } while (ReduceSendNotRecv && dst < nRails-1);
    }
  };

  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    const int part = ncclShmem.channelId - work->channelLo;
    const int nChannels = work->channelHi - work->channelLo + 1;
    struct ncclDirect* direct = &ncclShmem.channel.collnetDirect;
    int const &nNodes = ncclShmem.comm.nNodes;
    ssize_t chunkSize = int(work->collnet.chunkCount);
    ssize_t countPerRank = work->collnet.count;
    const int hasDn = (direct->down[0] >= 0) ? 1 : 0;

    if (direct->out == -1) __trap();
    bool isMultiRail = (direct->nHeads > 1);
    int nWarps1 = (isMultiRail ? 2 : 0);
    int nWarps2 = (isMultiRail ? 2 : 1);
    int nWarps3 = 1;
    float denom = float(work->nWarps)/float(nWarps1+nWarps2+nWarps3);
    nWarps3 = int(denom*nWarps3);
    nWarps2 = int(denom*nWarps2);
    nWarps1 = work->nWarps - (nWarps2+nWarps3);

    using Proto = ProtoSimple<1, 1>;

    int tn = nWarps1*WARP_SIZE;
    if (tid < tn) {
      // Phase 1: Scatter inputs to peers
      Primitives<T, RedOp, FanAsymmetric<0, NCCL_MAX_DIRECT_ARITY>, /*Direct=*/0, Proto, 0>
        prims(tid, tn, nullptr, direct->heads+1, work->sendbuff, nullptr,
              work->redOpArg, 0*Proto::MaxGroupWidth, 1, 1);
      for (ssize_t railGridOffset=0; railGridOffset < nNodes*countPerRank; railGridOffset += nChannels*chunkSize) {
        Scatterer</*ReduceSendNotRecv=*/true> scat;
        scat.work = work;
        scat.chunkSize = chunkSize;
        scat.railGridOffset = railGridOffset;
        prims.template process</*Recv=*/0, /*Send=*/1>(scat, 0, 0);
      }
      return;
    }
    tid -= tn;

    tn = nWarps2*WARP_SIZE;
    if (tid < tn) {
      if (work->netRegUsed && !hasDn) {
        if (tid == 0) {
          Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_DIRECT_ARITY, 1>, /*Direct=*/0, Proto, 0>::sendPeerNotify(direct->out, 1, 1);
        }
        __syncwarp();
      } else {
        // Phase 2: Reduce from peers + local input -> send to network
        Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_DIRECT_ARITY, 1>, /*Direct=*/0, Proto, 0>
          prims(tid, tn, direct->heads + 1, &direct->out, nullptr, nullptr,
            work->redOpArg, 1 * Proto::MaxGroupWidth, 1, 1);
        for (ssize_t railGridOffset = 0; railGridOffset < nNodes * countPerRank; railGridOffset += nChannels * chunkSize) {
          Scatterer</*ReduceSendNotRecv=*/false> scat;
          scat.work = work;
          scat.chunkSize = chunkSize;
          scat.railGridOffset = railGridOffset;
          prims.template process</*Recv=*/1, /*Send=*/1>(scat, 0, 0);
        }
      }
      return;
    }
    tid -= tn;

    tn = nWarps3*WARP_SIZE;
    if (tid < tn) {
      if (work->netRegUsed) {
        if (tid == 0) {
          int steps = hasDn ? (int)divUp(nNodes * countPerRank, nChannels * chunkSize) : 1;
          Primitives<T, RedOp, FanAsymmetric<1, 0>, /*Direct=*/0, Proto, 0>::recvPeerNotify(direct->out, 0, steps);
        }
        __syncwarp();
      } else {
        // Phase 3: recv from network
        Primitives<T, RedOp, FanAsymmetric<1, 0>, /*Direct=*/0, Proto, 0>
          prims(tid, tn, &direct->out, nullptr, nullptr, work->recvbuff,
            work->redOpArg, 2 * Proto::MaxGroupWidth, 0, 0);
        for (ssize_t railGridOffset = 0; railGridOffset < nNodes * countPerRank; railGridOffset += nChannels * chunkSize) {
          ssize_t railAllBeg = railGridOffset + part * chunkSize;
          ssize_t railAllEnd = min(railAllBeg + chunkSize, nNodes * countPerRank);
          ssize_t railOneBeg = ncclShmem.comm.node * countPerRank;
          ssize_t railOneEnd = railOneBeg + countPerRank;
          ssize_t beg = max(railAllBeg, railOneBeg);
          ssize_t end = min(railAllEnd, railOneEnd);
          prims.recv(beg - railOneBeg, max(ssize_t(0), end - beg), /*postOp=*/true);
        }
      }
      return;
    }
  }
};

template <typename T, typename RedOp>
struct RunWorkColl<ncclFuncReduceScatter, T, RedOp, NCCL_ALGO_BINE, NCCL_PROTO_SIMPLE>
{
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl *work)
  {
    ncclBine *bine = &ncclShmem.channel.bine;
    const ncclBineBufferManagement_t bufferManagement = bine->bufferManagement;
    const ncclBineVariant_t bineVar = bine->variant;

    using Proto = ProtoSimple<1, 1>;

    if (bineVar == TREE) {
      runBineTree<T, RedOp, Proto>(tid, nthreads, work);
    }
    else if (bineVar == BUTTERFLY) {
      switch (bufferManagement) {
        case BLOCK_BY_BLOCK:
          runBineBlockByBlock<T, RedOp, Proto>(tid, nthreads, work);
          break;
        case PERMUTATION:
          runBinePermutation<T, RedOp, Proto>(tid, nthreads, work);
          break;
        case DOUBLE_SEND:
          runBineDoubleSend<T, RedOp, Proto>(tid, nthreads, work);
          break;
        case SEND:
          runBineSend<T, RedOp, Proto>(tid, nthreads, work);
          break;
        default:
          assert(false && "Invalid Bine buffer management");
          break;
      }
    }
    else {
      assert(false && "Invalid Bine variant");
    }
  }
};

template <typename T, typename RedOp>
struct RunWorkColl<ncclFuncReduceScatter, T, RedOp, NCCL_ALGO_BINE, NCCL_PROTO_LL>
{
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl *work)
  {
    ncclBine *bine = &ncclShmem.channel.bine;
    const ncclBineBufferManagement_t bufferManagement = bine->bufferManagement;
    const ncclBineVariant_t bineVar = bine->variant;

    if (bineVar == BUTTERFLY) {
      switch (bufferManagement) {
        case BLOCK_BY_BLOCK:
          runBineBlockByBlock<T, RedOp, ProtoLL>(tid, nthreads, work);
          break;
        case PERMUTATION:
          runBinePermutation<T, RedOp, ProtoLL>(tid, nthreads, work);
          break;
        case DOUBLE_SEND:
          runBineDoubleSend<T, RedOp, ProtoLL>(tid, nthreads, work);
          break;
        case SEND:
          runBineSend<T, RedOp, ProtoLL>(tid, nthreads, work);
          break;
        default:
          assert(false && "Invalid Bine buffer management");
          break;
      }
    }
    else {
      assert(false && "Invalid Bine variant");
    }
  }
};

template <typename T, typename RedOp>
struct RunWorkColl<ncclFuncReduceScatter, T, RedOp, NCCL_ALGO_BINE, NCCL_PROTO_LL128>
{
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl *work)
  {
    ncclBine *bine = &ncclShmem.channel.bine;
    const ncclBineBufferManagement_t bufferManagement = bine->bufferManagement;
    const ncclBineVariant_t bineVar = bine->variant;

    if (bineVar == BUTTERFLY) {
      switch (bufferManagement) {
        case BLOCK_BY_BLOCK:
          runBineBlockByBlock<T, RedOp, ProtoLL128>(tid, nthreads, work);
          break;
        case PERMUTATION:
          runBinePermutation<T, RedOp, ProtoLL128>(tid, nthreads, work);
          break;
        case DOUBLE_SEND:
          runBineDoubleSend<T, RedOp, ProtoLL128>(tid, nthreads, work);
          break;
        case SEND:
          runBineSend<T, RedOp, ProtoLL128>(tid, nthreads, work);
          break;
        default:
          assert(false && "Invalid Bine buffer management");
          break;
      }
    }
    else {
      assert(false && "Invalid Bine variant");
    }
  }
};
