/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "collectives.h"
#include "device.h"
#include "bine.h"
#include "primitives.h"
#include <cstdio>

namespace {
template <typename T, typename RedOp, typename Proto, bool isNetOffload = false>
__device__ __forceinline__ void runRing(int tid, int nthreads,
                                        struct ncclDevWorkColl *work) {
  ncclRing *ring = &ncclShmem.channel.ring;
  const int *ringRanks = ring->userRanks;
  const int nranks = ncclShmem.comm.nRanks;
  ssize_t count, partOffset, partCount, chunkCount;
  ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T), &count,
                  &partOffset, &partCount, &chunkCount);
  ssize_t offset;
  ssize_t dataOffset;
  int nelem;
  int rankDest;
  int workNthreads;
  T *inputBuf = (T *)work->sendbuff;
  T *outputBuf = (T *)work->recvbuff;

  // If isNetOffload == true, we only use 1 warp to drive Ring algo/network
  // communication and the rest of warps proceed to copy src data into dst
  // buffer in parallel when AG is not in-place.
  if (isNetOffload) {
    workNthreads = WARP_SIZE;
    chunkCount = NCCL_MAX_NET_SIZE;
  } else {
    workNthreads = nthreads;
  }

  if (tid < workNthreads) {
    // Coverity reports that the callee treats &ring->next as an array. However,
    // due to the use of FanSymmetric<1>, only the first element is ever
    // accessed, so it's fine. coverity[callee_ptr_arith:FALSE]
    Primitives<T, RedOp, FanSymmetric<1>, 1, Proto, 0, isNetOffload> prims(
        tid, workNthreads, &ring->prev, &ring->next, inputBuf, outputBuf,
        work->redOpArg, 0, 0, 0, work, NULL,
        isNetOffload ? NCCL_MAX_NET_SIZE : 0);
    for (size_t elemOffset = 0; elemOffset < partCount;
         elemOffset += chunkCount) {
      /////////////// begin AllGather steps ///////////////
      nelem = min(chunkCount, partCount - elemOffset);
      dataOffset = partOffset + elemOffset;

      // step 0: push data to next GPU
      rankDest = ringRanks[0];
      offset = dataOffset + rankDest * count;

      if ((inputBuf + dataOffset == outputBuf + offset) ||
          isNetOffload) { // In place or onePPN
        prims.directSend(dataOffset, offset, nelem);
      } else {
        prims.directCopySend(dataOffset, offset, nelem);
      }

      // k-2 steps: copy to next GPU
      for (int j = 1; j < nranks - 1; ++j) {
        rankDest = ringRanks[nranks - j];
        offset = dataOffset + rankDest * count;
        prims.directRecvCopyDirectSend(offset, offset, nelem);
      }

      // Make final copy from buffer to dest.
      rankDest = ringRanks[1];
      offset = dataOffset + rankDest * count;

      // Final wait/copy.
      prims.directRecv(offset, nelem);
    }
  } else if (inputBuf != outputBuf + ringRanks[0] * count) {
    inputBuf = inputBuf + partOffset;
    outputBuf = outputBuf + partOffset + ringRanks[0] * count;
    reduceCopy<COLL_UNROLL, RedOp, T, 0, 1, 1, 0, 1, 1, /*PreOpSrcs=*/0>(
        tid - workNthreads, nthreads - workNthreads, work->redOpArg, false, 1,
        (void **)&inputBuf, 1, (void **)&outputBuf, partCount);
  }
  // we have to wait for all warps before we can proceed to the next work;
  // otherwise, we can have contention if next work will use the outputBuf
  // in this work. We use bar 14 to avoid conflicts with prims barrier and
  // __syncthread().
  if (isNetOffload)
    barrier_sync(14, nthreads);
}

template <typename T, typename RedOp, typename Proto>
__device__ __forceinline__ void runBineSend(int tid, int nthreads,
                                            ncclDevWorkColl *work) {
  ncclBine *bine = &ncclShmem.channel.bine;
  const int steps = bine->nDoublingSteps;  // nRanks == 1 << steps

  ssize_t count, gridOffset, channelCount, chunkCount;
  ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T), &count,
                  &gridOffset, &channelCount, &chunkCount);

  T *recvBuf = (T *)work->recvbuff;
  T *sendBuf = (T *)work->sendbuff;
  const int rank = ncclShmem.comm.rank;

  // SEND views over the generic tables.
  const int myIdx      = bineSendIndex(bine->index, rank, steps);  // position I own
  const int redistTo   = bineSendOrder(bine->order, rank, steps);  // rank that owns position `rank`
  const int redistFrom = myIdx;                                    // I need block myIdx (from rank myIdx)

  const bool useDirect = (work->direct & (NCCL_P2P_READ | NCCL_P2P_WRITE)) ==
                         (NCCL_P2P_READ | NCCL_P2P_WRITE);

  // ---- 1. Pre-step: put block `myIdx` at position `myIdx` ----
  T *mySeg = recvBuf + (ssize_t)rank * count;
  if (redistTo == rank) {
    if (sendBuf != mySeg) {
      for (ssize_t elem = tid; elem < channelCount; elem += nthreads)
        mySeg[gridOffset + elem] = sendBuf[gridOffset + elem];
    }
  } else {
    int sendPeer[1] = {redistTo};
    int recvPeer[1] = {redistFrom};

    // My block must land at offset `rank` in redistTo's buffer, because
    // index_send[redistTo] == rank. It is NOT redistTo * count.
    const ssize_t dstOffForPeer = (ssize_t)rank * count + gridOffset;
    const ssize_t myOff         = (ssize_t)myIdx * count + gridOffset;

    // One Primitives for the whole pre-step. Send and recv go chunk by chunk,
    // so we never have more than one chunk in flight before receiving.
    auto pre = [&](auto &prim, bool direct) {
      for (ssize_t elem = 0; elem < channelCount; elem += chunkCount) {
        const int ne = (int)min(chunkCount, channelCount - elem);
        if (direct) {
          prim.directSend(gridOffset + elem, dstOffForPeer + elem, ne);
          prim.directRecv(myOff + elem, ne);
        } else {
          prim.send(gridOffset + elem, ne);
          prim.recv(myOff + elem, ne);
        }
      }
    };

    if (useDirect) {
      Primitives<T, RedOp, FanAsymmetric<1, 1>, 1, Proto, 0> prim(
          tid, nthreads, recvPeer, sendPeer, sendBuf, recvBuf, work->redOpArg);
      pre(prim, true);
    } else {
      Primitives<T, RedOp, FanAsymmetric<1, 1>, 0, Proto, 0> prim(
          tid, nthreads, recvPeer, sendPeer, sendBuf, recvBuf, work->redOpArg);
      pre(prim, false);
    }
  }
  __syncthreads();  // positions must be in place before step 0 reads them

  // ---- 2. Distance-doubling butterfly: 1 Primitives per step ----
  for (int s = 0; s < steps; ++s) {
    const int partner = bineSendPartner(bine->partners, rank, s, steps);
    if (partner < 0) continue;

    const int span       = 1 << s;
    const int blockSize  = span << 1;
    const int base       = (myIdx / blockSize) * blockSize;
    const bool keepLower = ((myIdx >> s) & 1) == 0;

    // I hold the half I keep and send it; I receive the other half.
    const int sendBeg = keepLower ? base : base + span;
    const int recvBeg = keepLower ? base + span : base;

    int peers[1] = {partner};

    auto run = [&](auto &prim, bool direct) {
      if (channelCount == count) {
        // This channel covers whole blocks, so the span is one contiguous range.
        const ssize_t sBase = (ssize_t)sendBeg * count + gridOffset;
        const ssize_t rBase = (ssize_t)recvBeg * count + gridOffset;
        const ssize_t total = (ssize_t)span * count;
        // Pieces must fit in the FIFO without the peer receiving, or both sides
        // block in send. Keep them aligned to chunkCount. Start at 2*chunkCount, then tune.
        const ssize_t piece = chunkCount * 2;
        for (ssize_t o = 0; o < total; o += piece) {
          const ssize_t n = min(piece, total - o);
          if (direct) { prim.directSendFromOutput(sBase + o, n); prim.directRecv(rBase + o, n); }
          else        { prim.sendFromOutput(sBase + o, n);       prim.recv(rBase + o, n); }
        }
      } else {
        // Channel slices break contiguity: per-block, per-chunk (your current loop).
        for (ssize_t elem = 0; elem < channelCount; elem += chunkCount) {
          const ssize_t dataOff = gridOffset + elem;
          const int ne = (int)min(chunkCount, channelCount - elem);
          for (int j = 0; j < span; ++j) {
            const ssize_t sOff = (ssize_t)(sendBeg + j) * count + dataOff;
            const ssize_t rOff = (ssize_t)(recvBeg + j) * count + dataOff;
            if (direct) { prim.directSendFromOutput(sOff, ne); prim.directRecv(rOff, ne); }
            else        { prim.sendFromOutput(sOff, ne);       prim.recv(rOff, ne); }
          }
        }
      }
    };

    if (useDirect) {
      Primitives<T, RedOp, FanAsymmetric<1, 1>, 1, Proto, 0> prim(
          tid, nthreads, peers, peers, recvBuf, recvBuf, work->redOpArg);
      run(prim, true);
    } else {
      Primitives<T, RedOp, FanAsymmetric<1, 1>, 0, Proto, 0> prim(
          tid, nthreads, peers, peers, recvBuf, recvBuf, work->redOpArg);
      run(prim, false);
    }
    __syncthreads();  // step s+1 reads data received in step s
  }
}


template <typename T, typename RedOp, typename Proto>
__device__ __forceinline__ void runBineDoubleSend(int tid, int nthreads,
                                                    ncclDevWorkColl *work) {
  ncclBine *bine = &ncclShmem.channel.bine;
  const int steps = bine->nDoublingSteps;  // nRanks == 1 << steps

  ssize_t count, gridOffset, channelCount, chunkCount;
  ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T), &count,
                  &gridOffset, &channelCount, &chunkCount);
  if (channelCount == 0) return;

  T *recvBuf = (T *)work->recvbuff;
  T *sendBuf = (T *)work->sendbuff;
  const int p    = ncclShmem.comm.nRanks;
  const int rank = ncclShmem.comm.rank;
  const bool useDirect = (work->direct & (NCCL_P2P_READ | NCCL_P2P_WRITE)) ==
                         (NCCL_P2P_READ | NCCL_P2P_WRITE);

  // ---- 1. Own block to its natural slot (local copy) ----
  T *mySeg = recvBuf + (ssize_t)rank * count;
  if (sendBuf != mySeg) {
    for (ssize_t e = tid; e < channelCount; e += nthreads)
      mySeg[gridOffset + e] = sendBuf[gridOffset + e];
  }
  __syncthreads();  // slot `rank` must be filled before the first send reads it
  if (steps == 0 || !bine->dhlvPartners) return;

  // ---- 2. Reverse of the RS schedule: window start `a`, size 1 << g ----
  int a = rank;  // RS always ends with window [rank]; AG starts there

  for (int s = steps - 1; s >= 0; --s) {
    const int g    = steps - 1 - s;
    const int span = 1 << g;
    const bool sendBottom = ((rank & 1) == (g & 1));  // same rule as the RS kernel

    // I hold [a, a+span) and send it. The RS sent the other half of the doubled window.
    const int sendStart = a;
    const int recvStart = sendBottom ? (a - span + p) % p : (a + span) % p;
    if (sendBottom) a = recvStart;  // window grows downward; otherwise start unchanged

    const int partner = bine->dhlvPartners[rank * steps + s];
    if (partner >= 0 && partner != rank) {
      int peers[1] = {partner};

      auto run = [&](auto &prim, bool direct) {
        for (ssize_t elem = 0; elem < channelCount; elem += chunkCount) {
          const ssize_t dataOff = gridOffset + elem;
          const int ne = (int)min(chunkCount, channelCount - elem);
          for (int j = 0; j < span; ++j) {
            const ssize_t sOff = (ssize_t)((sendStart + j) % p) * count + dataOff;
            const ssize_t rOff = (ssize_t)((recvStart + j) % p) * count + dataOff;
            if (direct) { prim.directSendFromOutput(sOff, ne); prim.directRecv(rOff, ne); }
            else        { prim.sendFromOutput(sOff, ne);       prim.recv(rOff, ne); }
          }
        }
      };

      if (useDirect) {
        Primitives<T, RedOp, FanAsymmetric<1, 1>, 1, Proto, 0> prim(
            tid, nthreads, peers, peers, recvBuf, recvBuf, work->redOpArg,
            0, 0, 0, work);
        run(prim, true);
      } else {
        Primitives<T, RedOp, FanAsymmetric<1, 1>, 0, Proto, 0> prim(
            tid, nthreads, peers, peers, recvBuf, recvBuf, work->redOpArg,
            0, 0, 0, work);
        run(prim, false);
      }
    }
    __syncthreads();  // next step sends blocks received in this one
  }
}

template <typename T, typename RedOp, typename Proto>
__device__ __forceinline__ void runBineBlockByBlock(int tid, int nthreads,
                                                    ncclDevWorkColl *work) {
  ncclBine *bine = &ncclShmem.channel.bine;
  const int steps = bine->nDoublingSteps;

  ssize_t count, gridOffset, channelCount, chunkCount;
  ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T), &count,
                  &gridOffset, &channelCount, &chunkCount);
  if (channelCount == 0) return;

  T *recvBuf = (T *)work->recvbuff;
  T *sendBuf = (T *)work->sendbuff;
  const int rank = ncclShmem.comm.rank;

  T *mySeg = recvBuf + (ssize_t)rank * count;
  if (sendBuf != mySeg) {
    for (ssize_t e = tid; e < channelCount; e += nthreads)
      mySeg[gridOffset + e] = sendBuf[gridOffset + e];
  }
  __syncthreads();

  const int myIdx = bineSendIndex(bine->index, rank, steps);
  const bool useDirect = (work->direct & (NCCL_P2P_READ | NCCL_P2P_WRITE)) ==
                         (NCCL_P2P_READ | NCCL_P2P_WRITE);

  for (int s = 0; s < steps; ++s) {
    const int partner = bineSendPartner(bine->partners, rank, s, steps);
    if (partner >= 0) {
      const int span = 1 << s;
      const int base = (myIdx >> (s + 1)) << (s + 1);
      const bool keepLower = ((myIdx >> s) & 1) == 0;
      const int sendBeg = keepLower ? base : base + span;
      const int recvBeg = keepLower ? base + span : base;
      int peers[1] = {partner};

      auto run = [&](auto &prim, bool direct) {
        for (ssize_t elem = 0; elem < channelCount; elem += chunkCount) {
          const ssize_t dataOff = gridOffset + elem;
          const int ne = (int)min(chunkCount, channelCount - elem);
          for (int j = 0; j < span; ++j) {
            // position -> block id (= rank that owns that position)
            const int sBlk = bineSendOrder(bine->order, sendBeg + j, steps);
            const int rBlk = bineSendOrder(bine->order, recvBeg + j, steps);
            const ssize_t sOff = (ssize_t)sBlk * count + dataOff;
            const ssize_t rOff = (ssize_t)rBlk * count + dataOff;
            if (direct) { prim.directSendFromOutput(sOff, ne); prim.directRecv(rOff, ne); }
            else        { prim.sendFromOutput(sOff, ne);       prim.recv(rOff, ne); }
          }
        }
      };

      if (useDirect) {
        Primitives<T, RedOp, FanAsymmetric<1, 1>, 1, Proto, 0> prim(
            tid, nthreads, peers, peers, recvBuf, recvBuf, work->redOpArg);
        run(prim, true);
      } else {
        Primitives<T, RedOp, FanAsymmetric<1, 1>, 0, Proto, 0> prim(
            tid, nthreads, peers, peers, recvBuf, recvBuf, work->redOpArg);
        run(prim, false);
      }
    }
    __syncthreads();
  }
}

// In-place: new[x] = old[f(x)], restricted to this channel's column slice.
// f must be a permutation of [0, nRanks). Cycle-following, no scratch buffer.
template <typename T, typename F>
__device__ __forceinline__ void bineInPlacePermute(
    int tid, int nthreads, T *buf, F f, int nRanks,
    ssize_t count, ssize_t gridOffset, ssize_t channelCount)
{
  for (int i = 0; i < nRanks; ++i) {
    int j = f(i);
    if (j == i) continue;
    bool leader = true;                       // is i the smallest index in its cycle?
    while (j != i) {
      if (j < i) { leader = false; break; }
      j = f(j);
    }
    if (!leader) continue;

    for (ssize_t e = tid; e < channelCount; e += nthreads) {
      const ssize_t col = gridOffset + e;
      T tmp = buf[(ssize_t)i * count + col];
      int cur = i;
      while (true) {
        const int nxt = f(cur);
        if (nxt == i) { buf[(ssize_t)cur * count + col] = tmp; break; }
        buf[(ssize_t)cur * count + col] = buf[(ssize_t)nxt * count + col];
        cur = nxt;
      }
    }
  }
}

template <typename T, typename RedOp, typename Proto>
__device__ __forceinline__ void runBinePermutation(int tid, int nthreads,
                                                     ncclDevWorkColl *work) {
  ncclBine *bine = &ncclShmem.channel.bine;
  const int steps = bine->nDoublingSteps;  // nRanks == 1 << steps

  ssize_t count, gridOffset, channelCount, chunkCount;
  ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T), &count,
                  &gridOffset, &channelCount, &chunkCount);
  if (channelCount == 0) return;

  T *recvBuf = (T *)work->recvbuff;
  T *sendBuf = (T *)work->sendbuff;
  const int nRanks = ncclShmem.comm.nRanks;
  const int rank   = ncclShmem.comm.rank;
  const int myIdx  = bineSendIndex(bine->index, rank, steps);  // position I own

  const bool useDirect = (work->direct & (NCCL_P2P_READ | NCCL_P2P_WRITE)) ==
                         (NCCL_P2P_READ | NCCL_P2P_WRITE);

  // ---- 1. Place my block at position myIdx (local copy, no communication) ----
  T *mySeg = recvBuf + (ssize_t)myIdx * count;
  if (sendBuf != mySeg) {
    for (ssize_t e = tid; e < channelCount; e += nthreads)
      mySeg[gridOffset + e] = sendBuf[gridOffset + e];
  }
  __syncthreads();  // position myIdx must be filled before step 0 sends it

  // ---- 2. Distance-doubling butterfly in position space ----
  for (int s = 0; s < steps; ++s) {
    const int partner = bineSendPartner(bine->partners, rank, s, steps);
    if (partner < 0) continue;

    const int span       = 1 << s;
    const int base       = (myIdx >> (s + 1)) << (s + 1);
    const bool keepLower = ((myIdx >> s) & 1) == 0;
    // I hold the half I keep and send it; I receive the other half.
    const int sendBeg = keepLower ? base : base + span;
    const int recvBeg = keepLower ? base + span : base;

    int peers[1] = {partner};

    auto run = [&](auto &prim, bool direct) {
      if (channelCount == count) {
        const ssize_t sBase = (ssize_t)sendBeg * count + gridOffset;
        const ssize_t rBase = (ssize_t)recvBeg * count + gridOffset;
        const ssize_t total = (ssize_t)span * count;
        const ssize_t piece = chunkCount * 2;   // same tuning knob as runBineSend
        for (ssize_t o = 0; o < total; o += piece) {
          const ssize_t n = min(piece, total - o);
          if (direct) { prim.directSendFromOutput(sBase + o, n); prim.directRecv(rBase + o, n); }
          else        { prim.sendFromOutput(sBase + o, n);       prim.recv(rBase + o, n); }
        }
      } else {
        for (ssize_t elem = 0; elem < channelCount; elem += chunkCount) {
          const ssize_t dataOff = gridOffset + elem;
          const int ne = (int)min(chunkCount, channelCount - elem);
          for (int j = 0; j < span; ++j) {
            const ssize_t sOff = (ssize_t)(sendBeg + j) * count + dataOff;
            const ssize_t rOff = (ssize_t)(recvBeg + j) * count + dataOff;
            if (direct) { prim.directSendFromOutput(sOff, ne); prim.directRecv(rOff, ne); }
            else        { prim.sendFromOutput(sOff, ne);       prim.recv(rOff, ne); }
          }
        }
      }
    };

    if (useDirect) {
      Primitives<T, RedOp, FanAsymmetric<1, 1>, 1, Proto, 0> prim(
          tid, nthreads, peers, peers, recvBuf, recvBuf, work->redOpArg);
      run(prim, true);
    } else {
      Primitives<T, RedOp, FanAsymmetric<1, 1>, 0, Proto, 0> prim(
          tid, nthreads, peers, peers, recvBuf, recvBuf, work->redOpArg);
      run(prim, false);
    }
    __syncthreads();  // step s+1 reads data received in step s
  }

  // ---- 3. Final in-place permutation: block b moves from position index(b) to b ----
  const int *idxTbl = bine->index;
  bineInPlacePermute<T>(tid, nthreads, recvBuf,
      [&](int b) { return bineSendIndex(idxTbl, b, steps); },
      nRanks, count, gridOffset, channelCount);
}
} // namespace

template <typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllGather, T, RedOp, NCCL_ALGO_RING,
                   NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int nthreads,
                                      struct ncclDevWorkColl *work) {
    bool isNetOffload = work->isOneRPN && work->netRegUsed;
    if (isNetOffload)
      runRing<T, RedOp, ProtoSimple<1, 1>, true>(tid, nthreads, work);
    else
      runRing<T, RedOp,
              ProtoSimple<ALLGATHER_CHUNKSTEPS / ALLGATHER_SLICESTEPS,
                          ALLGATHER_SLICESTEPS>,
              false>(tid, nthreads, work);
  }
};

template <typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllGather, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL> {
  __device__ __forceinline__ void run(int tid, int nthreads,
                                      struct ncclDevWorkColl *work) {
    runRing<T, RedOp, ProtoLL>(tid, nthreads, work);
  }
};

template <typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllGather, T, RedOp, NCCL_ALGO_RING,
                   NCCL_PROTO_LL128> {
  __device__ __forceinline__ void run(int tid, int nthreads,
                                      struct ncclDevWorkColl *work) {
    runRing<T, RedOp, ProtoLL128>(tid, nthreads, work);
  }
};

template <typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllGather, T, RedOp, NCCL_ALGO_PAT,
                   NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int nthreads,
                                      struct ncclDevWorkColl *work) {
#if __CUDA_ARCH__ >= 600
    using Proto = ProtoSimple<1, 1>;
    const int nranks = ncclShmem.comm.nRanks;
    const int rank = ncclShmem.comm.rank;
    size_t count, channelOffset, channelCount, chunkCount;
    ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T), &count,
                    &channelOffset, &channelCount, &chunkCount);

    static constexpr int nworkers = NCCL_PAT_NWORKERS;
    struct ncclPatShmem *shmem = (struct ncclPatShmem *)ncclScratchForWarp(0);
    uint64_t pollCount = 0;
    __syncthreads(); // Don't start using shared mem until everyone arrives
    for (int i = tid; i < NCCL_SHMEM_PAT_STEPS; i += nthreads)
      shmem->patSteps[i].flags = 0;
    if (tid == 0)
      shmem->localAccSize = 0;
    if (tid == nworkers)
      shmem->parallelFactor = 0;
    __syncthreads();

    if (tid == nworkers) { // Algo computation thread
      PatAGAlgorithm<T> patAlgo(chunkCount * sizeof(T), NCCL_STEPS,
                                NCCL_PAT_NWORKERS / WARP_SIZE, channelOffset,
                                channelOffset + channelCount, count, chunkCount,
                                rank, nranks);
      int parallelFactor = shmem->parallelFactor = patAlgo.getParallelFactor();
      int step = 0;
      while (1) {
        struct ncclPatStep *ps =
            shmem->patSteps + (step % NCCL_SHMEM_PAT_STEPS);
        cuda::atomic_ref<int, cuda::thread_scope_block> poll(ps->flags);
        while (poll.load(cuda::memory_order_acquire) != 0)
          pollCount++; // Wait for workers to be done with step
                       // 'step-NCCL_SHMEM_PAT_STEPS'
        patAlgo.getNextOp(ps);
        int last = ps->last;
        step++;
        if (last == 2)
          break;
      }
    } else if (tid < nworkers) { // Worker threads
      T *inputBuf = (T *)work->sendbuff;
      T *outputBuf = (T *)work->recvbuff;
      int parallelFactor = 0;
      volatile int *pfPtr = &shmem->parallelFactor;
      while (parallelFactor == 0)
        parallelFactor = *pfPtr;

      int groupSize = nworkers / (WARP_SIZE * parallelFactor) * WARP_SIZE;
      int group = tid / groupSize;
      int nGroups = nworkers / groupSize;
      int tidInGroup = tid - group * groupSize;
      // We don't use recvPeers/sendPeers so let's pass shmem structs instead
      Primitives<T, RedOp, FanSymmetric<1>, 0, Proto, 0> prims(
          tidInGroup, groupSize, (int *)shmem->recvDims, (int *)shmem->sendDims,
          inputBuf, outputBuf, work->redOpArg, group, 0, 0, nullptr, nullptr, 0,
          primsModePatAg);

      int step = group;
      while (1) {
        struct ncclPatStep *ps =
            shmem->patSteps + (step % NCCL_SHMEM_PAT_STEPS);
        cuda::atomic_ref<int, cuda::thread_scope_block> poll(ps->flags);
        while (poll.load(cuda::memory_order_acquire) == 0)
          pollCount++; // Wait for compute thread
        int last = ps->last;
        prims.patCopy(ps, shmem);
        if (tidInGroup == 0)
          poll.store(
              0,
              cuda::memory_order_release); // Return element to compute thread
        if (last)
          break;
        step += nGroups;
      }
    }
#endif
  }
};

template <typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllGather, T, RedOp, NCCL_ALGO_NVLS,
                   NCCL_PROTO_SIMPLE> {
  template <bool BcastSendNotRecv> struct Scatterer {
    struct ncclDevWorkColl *work;
    ssize_t chunkSize;
    ssize_t railGridOffset;

    template <int SlicePerChunk, int MinSrcs, int MaxSrcs, int MinDsts,
              int MaxDsts, int MultimemSrcs, int MultimemDsts>
    __device__ __forceinline__ void
    operator()(int tid, int tn, int slice, int maxSliceSize, int nSrcs,
               void **srcPtrs, int nDsts, void **dstPtrs, int32_t *dstSizes,
               uint32_t sendDirectFlag, uint32_t recvDirectFlag) {
      static_assert(SlicePerChunk == 1, "require: SlicePerChunk==1");
      static_assert(MaxDsts <= 1 || MaxSrcs <= 1,
                    "require: MaxDsts<=1 || MaxSrcs<=1");

      struct ncclNvls *nvls = &ncclShmem.channel.nvls;
      int nNodes = ncclShmem.comm.nNodes;
      int nRails = nvls->nHeads;
      int part = ncclShmem.channelId - work->channelLo;
      char *inbuf = (char *)work->sendbuff;
      char *outbuf = (char *)work->recvbuff;
      ssize_t countPerRank = work->collnet.count;
      bool inPlace = (inbuf == outbuf + ncclShmem.comm.rank * countPerRank);
      ssize_t railAllBeg =
          min(railGridOffset + part * chunkSize, nNodes * countPerRank);
      ssize_t railAllEnd = min(railAllBeg + chunkSize, nNodes * countPerRank);
      int railAllSize = railAllEnd - railAllBeg;
      int rail = 0;
      int src = 0;

      if (BcastSendNotRecv) {
        rail = nvls->headRank;
      } else {
        if (work->regUsed)
          return;
        rail = 0;
      }
      if (tid < nDsts)
        dstSizes[tid] = railAllSize;
      do {
        int node = railAllBeg / countPerRank;
        int railAllOffset = 0;
        while (railAllOffset < railAllSize) {
          ssize_t railOneBeg = node * countPerRank;
          ssize_t railOneEnd = railOneBeg + countPerRank;
          ssize_t railOneOffset = (railAllBeg + railAllOffset) - railOneBeg;
          int delta =
              min(railAllEnd, railOneEnd) - (railAllBeg + railAllOffset);
          int rank =
              ncclShmem.comm.collNetDenseToUserRank[node * nRails + rail];
          ssize_t userOneBeg = rank * countPerRank + railOneOffset;
          int outIsDst = (inPlace && rank == ncclShmem.comm.rank) ||
                                 BcastSendNotRecv || work->regUsed
                             ? 0
                             : 1;
          if (nSrcs != 0 && outIsDst + nDsts != 0) {
            reduceCopy<ncclCollUnroll(), RedOp, T,
                       /*MultimemSrcs,MinSrcs,MaxSrcs=*/MultimemSrcs, 1, 1,
                       /*MultimemDsts=*/MultimemDsts,
                       0 + MultimemDsts + MinDsts, 1 + MaxDsts,
                       /*PreOpSrcs=*/0>(
                tid, tn, 0, false,
                /*nSrcs=*/1,
                [=] __device__(int s /*==0*/) -> void * {
                  return (char *)srcPtrs[src] + railAllOffset;
                },
                /*nDsts=*/outIsDst + nDsts,
                [=] __device__(int d) -> void * {
                  return d < outIsDst ? outbuf + userOneBeg
                         : work->regUsed
                             ? (char *)dstPtrs[d - outIsDst] + userOneBeg
                             : (char *)dstPtrs[d - outIsDst] + railAllOffset;
                },
                delta);
          }
          railAllOffset += delta;
          node += 1;
        }
        rail += 1;
        src += 1;
      } while (!BcastSendNotRecv && src < nRails);
    }
  };

  __device__ __forceinline__ void run(int tid, int /*nthreads*/,
                                      struct ncclDevWorkColl *work) {
    struct ncclNvls *nvls = &ncclShmem.channel.nvls;
    int nelem;

    const int nThreadsNetSend =
        work->oneNode ? 0 : (work->netRegUsed ? WARP_SIZE : 6 * WARP_SIZE);
    const int nThreadsGather =
        work->regUsed ? roundUp(nvls->nHeads << 2, WARP_SIZE) : 8 * WARP_SIZE;
    const int nThreadsBcast =
        NCCL_MAX_NTHREADS - nThreadsNetSend - nThreadsGather;

    const int tidEndGather = nThreadsGather;
    const int tidEndNetSend = tidEndGather + nThreadsNetSend;
    const int tidEndBcast = tidEndNetSend + nThreadsBcast;

    if (work->oneNode) {
      const ssize_t rank = ncclShmem.comm.rank;
      size_t count, gridOffset, channelCount, offset, chunkCount;
      ncclCollCbdPart(work, ncclShmem.channelId, NCCL_PROTO_SIMPLE, sizeof(T),
                      &count, &gridOffset, &channelCount, &chunkCount);
      if (!work->regUsed) {
        if (tid < tidEndGather) {
          // Gather
          using Proto = ProtoSimple<1, 1, COLL_UNROLL>;
          Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_NVLS_ARITY, 0>,
                     /*Direct=*/0, Proto, 0>
              prims(tid, nThreadsGather, nvls->up, NULL, NULL, work->recvbuff,
                    work->redOpArg, 0 * Proto::MaxGroupWidth, 1, 1);
          for (size_t elemOffset = 0; elemOffset < channelCount;
               elemOffset += chunkCount) {
            offset = gridOffset + elemOffset;
            nelem = min(chunkCount, channelCount - elemOffset);
            prims.gather(offset, nvls->nHeads * count, nelem, count, -1, 0);
          }
          // coverity[overrun-call] => Coverity think prims.index can be greater
          // than 1
        } else if (tid < tidEndBcast) {
          // Bcast through NVLS
          using Proto = ProtoSimple<1, 1, COLL_UNROLL, 0, 1>;
          Primitives<T, RedOp, FanAsymmetric<0, 1>, /*Direct=*/0, Proto, 0>
              prims(tid - tidEndGather, nThreadsBcast, NULL, &nvls->down,
                    work->sendbuff, NULL, work->redOpArg,
                    3 * Proto::MaxGroupWidth, 0, 0);
          for (size_t elemOffset = 0; elemOffset < channelCount;
               elemOffset += chunkCount) {
            offset = gridOffset + elemOffset;
            nelem = min(chunkCount, channelCount - elemOffset);
            prims.send(offset, nelem);
          }
          // coverity[overrun-call] => Coverity think prims.index can be greater
          // than 1
        }
      } else {
        if (tid < tidEndGather) {
          using Proto = ProtoSimple<1, 1, COLL_UNROLL>;
          Primitives<T, RedOp, FanSymmetric<NCCL_MAX_NVLS_ARITY>, /*Direct=*/0,
                     Proto, 0>
              prims(tid, nThreadsGather, nvls->up, nvls->up, NULL, NULL,
                    work->redOpArg, 0 * Proto::MaxGroupWidth, 1, 1);

          /* used as sync */
          prims.scatter(0, 0, 0, 0, -1, 0);

          for (size_t elemOffset = 0; elemOffset < channelCount;
               elemOffset += chunkCount) {
            prims.gather(0, 0, 0, 0, -1, 0);
          }
        } else if (tid < tidEndBcast) {
          using Proto = ProtoSimple<1, 1, COLL_UNROLL, 0, 1>;
          Primitives<T, RedOp, FanSymmetric<1>, /*Direct=*/1, Proto, 0> prims(
              tid - tidEndGather, nThreadsBcast, &nvls->down, &nvls->down,
              work->sendbuff, NULL, work->redOpArg, 1 * Proto::MaxGroupWidth, 0,
              0, work);
          /* used as sync */
          prims.recv(0, 0);

          for (size_t elemOffset = 0; elemOffset < channelCount;
               elemOffset += chunkCount) {
            ssize_t inpOffset = gridOffset + elemOffset;
            ssize_t outOffset = inpOffset + rank * count;
            nelem = min(chunkCount, channelCount - elemOffset);
            prims.directSend(inpOffset, outOffset, nelem);
          }
        }
      }
    } else {
      // NVLS + IB SHARP
      int nNodes = ncclShmem.comm.nNodes;
      int part = ncclShmem.channelId - work->channelLo;
      ssize_t countPerRank = work->collnet.count;
      const int nChannels = work->channelHi - work->channelLo + 1;
      ssize_t chunkCount = work->collnet.chunkCount;
      if (tid < tidEndGather) {
        using Proto = ProtoSimple<1, 1, COLL_UNROLL>;
        Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_NVLS_ARITY, 0>,
                   /*Direct=*/1, Proto, 0>
            prims(tid, nThreadsGather, nvls->up, nullptr, nullptr,
                  work->recvbuff,
                  /*redOpArg=*/0, 1 * Proto::MaxGroupWidth, 1, 1, work);
        for (ssize_t railGridOffset = 0; railGridOffset < nNodes * countPerRank;
             railGridOffset += nChannels * chunkCount) {
          Scatterer</*BcastSendNotRecv=*/false> scat;
          scat.work = work;
          scat.chunkSize = chunkCount;
          scat.railGridOffset = railGridOffset;
          prims.template process</*Recv=*/1, /*Send=*/0>(scat);
        }
      } else {
        if (work->netRegUsed) {
          using ProtoSend = ProtoSimple<1, 1, COLL_UNROLL>;
          using ProtoBcast = ProtoSimple<1, 1, COLL_UNROLL, 0, 1>;
          int maxSteps =
              (int)divUp(nNodes * countPerRank, nChannels * chunkCount);
          int curSteps = -1;
          int postThread = tid - tidEndGather == 0 ? 1 : 0;
          // for UB, we need to control the send speed to avoid net congestion.
          // first unroll 2 steps, then unroll the rest steps when the data is
          // received.
          if (postThread) {
            curSteps = min(2, maxSteps);
            Primitives<T, RedOp, FanAsymmetric<0, 1>, /*Direct=*/1, ProtoSend,
                       0>::sendPeerNotify(nvls->out, 1, curSteps);
          }
          Primitives<T, RedOp, FanSymmetric<1>, /*Direct=*/1, ProtoBcast, 0>
              prims(tid - tidEndGather, nThreadsNetSend + nThreadsBcast,
                    &nvls->out, &nvls->down, nullptr, nullptr,
                    /*redOpArg=*/0, 2 * ProtoBcast::MaxGroupWidth, 0, 0, work);
          for (ssize_t railGridOffset = 0;
               railGridOffset < nNodes * countPerRank;
               railGridOffset += nChannels * chunkCount) {
            Scatterer</*BcastSendNotRecv=*/true> scat;
            scat.work = work;
            scat.chunkSize = chunkCount;
            scat.railGridOffset = railGridOffset;
            prims.template process</*Recv=*/1, /*Send=*/1>(scat);
            if (postThread && curSteps < maxSteps) {
              curSteps++;
              Primitives<T, RedOp, FanAsymmetric<0, 1>, /*Direct=*/1, ProtoSend,
                         0>::sendPeerNotify(nvls->out, 1, 1);
            }
          }
        } else {
          if (tid < tidEndNetSend) {
            using Proto = ProtoSimple<1, 1, COLL_UNROLL>;
            Primitives<T, RedOp, FanAsymmetric<0, 1>, /*Direct=*/0, Proto, 0>
                prims(tid - tidEndGather, nThreadsNetSend, nullptr, &nvls->out,
                      work->sendbuff, nullptr,
                      /*redOpArg=*/0, 0 * Proto::MaxGroupWidth, 1, 1);
            for (ssize_t railGridOffset = 0;
                 railGridOffset < nNodes * countPerRank;
                 railGridOffset += nChannels * chunkCount) {
              ssize_t railAllBeg = railGridOffset + part * chunkCount;
              ssize_t railAllEnd =
                  min(railAllBeg + chunkCount, nNodes * countPerRank);
              ssize_t railOneBeg = ncclShmem.comm.node * countPerRank;
              ssize_t railOneEnd = railOneBeg + countPerRank;
              ssize_t beg = max(railAllBeg, railOneBeg);
              ssize_t end = min(railAllEnd, railOneEnd);
              prims.send(beg - railOneBeg, max(ssize_t(0), end - beg));
            }
          } else {
            using Proto = ProtoSimple<1, 1, COLL_UNROLL, 0, 1>;
            Primitives<T, RedOp, FanSymmetric<1>, /*Direct=*/0, Proto, 0> prims(
                tid - tidEndNetSend, nThreadsBcast, &nvls->out, &nvls->down,
                nullptr, nullptr,
                /*redOpArg=*/0, 2 * Proto::MaxGroupWidth, 0, 0);
            for (ssize_t railGridOffset = 0;
                 railGridOffset < nNodes * countPerRank;
                 railGridOffset += nChannels * chunkCount) {
              Scatterer</*BcastSendNotRecv=*/true> scat;
              scat.work = work;
              scat.chunkSize = chunkCount;
              scat.railGridOffset = railGridOffset;
              prims.template process</*Recv=*/1, /*Send=*/1>(scat);
            }
          }
        }
      }
    }
  }
};

template <typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllGather, T, RedOp, NCCL_ALGO_COLLNET_DIRECT,
                   NCCL_PROTO_SIMPLE> {
  template <bool BcastSendNotRecv> struct Scatterer {
    struct ncclDevWorkColl *work;
    ssize_t chunkSize;
    ssize_t railGridOffset;

    template <int SlicePerChunk, int MinSrcs, int MaxSrcs, int MinDsts,
              int MaxDsts, int MultimemSrcs, int MultimemDsts>
    __device__ __forceinline__ void
    operator()(int tid, int tn, int slice, int maxSliceSize, int nSrcs,
               void **srcPtrs, int nDsts, void **dstPtrs, int32_t *dstSizes,
               uint32_t sendDirectFlag, uint32_t recvDirectFlag) {
      static_assert(SlicePerChunk == 1, "require: SlicePerChunk==1");
      static_assert(MaxDsts <= 1 || MaxSrcs <= 1,
                    "require: MaxDsts<=1 || MaxSrcs<=1");

      struct ncclDirect *direct = &ncclShmem.channel.collnetDirect;
      int nNodes = ncclShmem.comm.nNodes;
      int nRails = direct->nHeads;
      int part = ncclShmem.channelId - work->channelLo;
      char *inbuf = (char *)work->sendbuff;
      char *outbuf = (char *)work->recvbuff;
      ssize_t countPerRank = work->collnet.count * sizeof(T);
      bool inPlace = (inbuf == outbuf + ncclShmem.comm.rank * countPerRank);

      ssize_t railAllBeg =
          min(railGridOffset + part * chunkSize, nNodes * countPerRank);
      ssize_t railAllEnd = min(railAllBeg + chunkSize, nNodes * countPerRank);
      int railAllSize = railAllEnd - railAllBeg;
      if (tid < nDsts)
        dstSizes[tid] = railAllSize;

      int src = 0;
      int rail;
      if (BcastSendNotRecv) {
        rail = direct->headRank;
      } else {
        rail = direct->headRank + 1;
        if (rail == nRails)
          rail = 0;
      }
      do {
        int node = railAllBeg / countPerRank;
        int railAllOffset = 0;
        while (railAllOffset < railAllSize) {
          ssize_t railOneBeg = node * countPerRank;
          ssize_t railOneEnd = railOneBeg + countPerRank;
          ssize_t railOneOffset = (railAllBeg + railAllOffset) - railOneBeg;
          int delta =
              min(railAllEnd, railOneEnd) - (railAllBeg + railAllOffset);
          int rank =
              ncclShmem.comm.collNetDenseToUserRank[node * nRails + rail];
          ssize_t userOneBeg = rank * countPerRank + railOneOffset;
          int outIsDst = (inPlace && rank == ncclShmem.comm.rank) ? 0 : 1;
          if (nSrcs != 0 && outIsDst + nDsts != 0) {
            reduceCopy<ncclCollUnroll(), RedOp, T,
                       /*MultimemSrcs,MinSrcs,MaxSrcs=*/0, 1, 1,
                       /*MultimemDsts=*/0, 0 + MinDsts, 1 + MaxDsts,
                       /*PreOpSrcs=*/0>(
                tid, tn, 0, false,
                /*nSrcs=*/1,
                [=] __device__(int s /*==0*/) -> void * {
                  return work->regUsed && (recvDirectFlag & NCCL_P2P_READ)
                             ? (char *)srcPtrs[src] + userOneBeg
                             : (char *)srcPtrs[src] + railAllOffset;
                },
                /*nDsts=*/outIsDst + nDsts,
                [=] __device__(int d) -> void * {
                  return d < outIsDst ? outbuf + userOneBeg
                         : work->regUsed && (sendDirectFlag & NCCL_P2P_WRITE)
                             ? (char *)dstPtrs[d - outIsDst] + userOneBeg
                             : (char *)dstPtrs[d - outIsDst] + railAllOffset;
                },
                delta);
          }
          railAllOffset += delta;
          node += 1;
        }
        src += 1;
        rail += 1;
        if (rail == nRails)
          rail = 0;
      } while (!BcastSendNotRecv && src < nRails - 1);
    }
  };

  __device__ __forceinline__ void run(int tid, int /*nthreads*/,
                                      struct ncclDevWorkColl *work) {
    const int part = ncclShmem.channelId - work->channelLo;
    const int nChannels = work->channelHi - work->channelLo + 1;
    struct ncclDirect *direct = &ncclShmem.channel.collnetDirect;
    int const &nNodes = ncclShmem.comm.nNodes;
    ssize_t countPerRank = work->collnet.count;
    size_t chunkSize = work->collnet.chunkCount;
    const int hasDn = (direct->down[0] >= 0) ? 1 : 0;
    bool isMultiRail = (direct->nHeads > 1);
    int nWarps1 = 1;
    int nWarps2 = (isMultiRail ? 2 : 1);
    int nWarps3 = (isMultiRail ? 2 : 0);
    float denom = float(work->nWarps) / float(nWarps1 + nWarps2 + nWarps3);
    nWarps3 = int(denom * nWarps3);
    nWarps2 = int(denom * nWarps2);
    nWarps1 = work->nWarps - (nWarps2 + nWarps3);

    using Proto = ProtoSimple<1, 1>;

    int tn = nWarps1 * WARP_SIZE;
    if (tid < tn) {
      if (work->netRegUsed) {
        if (tid == 0) {
          // If this rank has local peers (i.e, hasDn == true), we cannot
          // offload all data to network. In this case, steps should be computed
          // based on chunkSize and so on; otherwise, we just bump the step by 1
          // to kick off collnet progress.
          int steps =
              hasDn ? (int)divUp(nNodes * countPerRank, nChannels * chunkSize)
                    : 1;
          Primitives<T, RedOp, FanAsymmetric<0, 1>, /*Direct=*/0, Proto,
                     0>::sendPeerNotify(direct->out, 1, steps);
        }
        __syncwarp();
      } else {
        // Phase 1: send to network
        Primitives<T, RedOp, FanAsymmetric<0, 1>, /*Direct=*/0, Proto, 0> prims(
            tid, tn, nullptr, &direct->out, work->sendbuff, nullptr,
            /*redOpArg=*/0, 0 * Proto::MaxGroupWidth, 1, 1);
        for (ssize_t railGridOffset = 0; railGridOffset < nNodes * countPerRank;
             railGridOffset += nChannels * chunkSize) {
          ssize_t railAllBeg = railGridOffset + part * chunkSize;
          ssize_t railAllEnd =
              min(railAllBeg + chunkSize, nNodes * countPerRank);
          ssize_t railOneBeg = ncclShmem.comm.node * countPerRank;
          ssize_t railOneEnd = railOneBeg + countPerRank;
          ssize_t beg = max(railAllBeg, railOneBeg);
          ssize_t end = min(railAllEnd, railOneEnd);
          prims.send(beg - railOneBeg, max(ssize_t(0), end - beg));
        }
      }
      return;
    }
    tid -= tn;

    tn = nWarps2 * WARP_SIZE;
    if (tid < tn) {
      if (work->netRegUsed && !hasDn) {
        if (tid == 0) {
          Primitives<T, RedOp, FanAsymmetric<1, NCCL_MAX_DIRECT_ARITY>,
                     /*Direct=*/0, Proto, 0>::recvPeerNotify(direct->out, 0, 1);
        }
        __syncwarp();
      } else {
        // Phase 2: Recv network -> deposit output + send to bcast
        Primitives<T, RedOp, FanAsymmetric<1, NCCL_MAX_DIRECT_ARITY>,
                   /*Direct=*/1, Proto, 0>
            prims(tid, tn, &direct->out, direct->heads + 1, nullptr,
                  work->recvbuff,
                  /*redOpArg=*/0, 1 * Proto::MaxGroupWidth, 0, 0, work);
        for (ssize_t railGridOffset = 0; railGridOffset < nNodes * countPerRank;
             railGridOffset += nChannels * chunkSize) {
          Scatterer</*BcastSendNotRecv=*/true> scat;
          scat.work = work;
          scat.chunkSize = chunkSize;
          scat.railGridOffset = railGridOffset;
          prims.template process</*Recv=*/1, /*Send=*/1>(scat, work->direct, 0);
        }
      }
      return;
    }
    tid -= tn;

    tn = nWarps3 * WARP_SIZE;
    if (tid < tn) {
      // Phase 3: Recv bcast -> deposit output
      Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_DIRECT_ARITY, 0>,
                 /*Direct=*/1, Proto, 0>
          prims(tid, tn, direct->heads + 1, nullptr, nullptr, work->recvbuff,
                /*redOpArg=*/0, 2 * Proto::MaxGroupWidth, 0, 0, work);
      for (ssize_t railGridOffset = 0; railGridOffset < nNodes * countPerRank;
           railGridOffset += nChannels * chunkSize) {
        Scatterer</*BcastSendNotRecv=*/false> scat;
        scat.work = work;
        scat.chunkSize = chunkSize;
        scat.railGridOffset = railGridOffset;
        prims.template process</*Recv=*/1, /*Send=*/0>(scat, 0, work->direct);
      }
      return;
    }
  }
};

template <typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllGather, T, RedOp, NCCL_ALGO_BINE,
                   NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int nthreads,
                                      struct ncclDevWorkColl *work) {
    ncclBine *bine = &ncclShmem.channel.bine;
    const ncclBineBufferManagement_t bufferManagement = bine->bufferManagement;

    // XXX: Temporary hack to fix Bine's chunkSteps=1 for allgather.
    using Proto = ProtoSimple<1,1>;

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
};

template <typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllGather, T, RedOp, NCCL_ALGO_BINE, NCCL_PROTO_LL> {
  __device__ __forceinline__ void run(int tid, int nthreads,
                                      struct ncclDevWorkColl *work) {
    ncclBine *bine = &ncclShmem.channel.bine;
    const ncclBineBufferManagement_t bufferManagement = bine->bufferManagement;

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
};

template <typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllGather, T, RedOp, NCCL_ALGO_BINE,
                   NCCL_PROTO_LL128> {
  __device__ __forceinline__ void run(int tid, int nthreads,
                                      struct ncclDevWorkColl *work) {
    ncclBine *bine = &ncclShmem.channel.bine;
    const ncclBineBufferManagement_t bufferManagement = bine->bufferManagement;

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
};
