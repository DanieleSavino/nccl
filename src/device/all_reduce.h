/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "device.h"
#include "bine.h"
#include "collectives.h"
#include "primitives.h"

namespace {
  template<typename T, typename RedOp, typename Proto>
  __device__ __forceinline__ void runRing(int tid, int nthreads, struct ncclDevWorkColl* work) {
    ncclRing *ring = &ncclShmem.channel.ring;
    int ringIx = ring->index;
    const int nranks = ncclShmem.comm.nRanks;
    ssize_t gridOffset;
    ssize_t channelCount;
    ssize_t chunkCount;
    ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T), (ssize_t*)nullptr, &gridOffset, &channelCount, &chunkCount);
    const ssize_t loopCount = nranks * chunkCount;
    ssize_t offset;
    int nelem;
    int chunk;

    // Coverity reports that the callee treats &ring->next as an array.  However, due to the use of
    // FanSymmetric<1>, only the first element is ever accessed, so it's fine.
    // coverity[callee_ptr_arith:FALSE]
    Primitives<T, RedOp, FanSymmetric<1>, 1, Proto, 0> prims
      (tid, nthreads, &ring->prev, &ring->next, work->sendbuff, work->recvbuff, work->redOpArg, 0, 0, 0, work);

    for (ssize_t elemOffset = 0; elemOffset < channelCount; elemOffset += loopCount) {
      ssize_t remCount = channelCount - elemOffset;
      ssize_t chunkOffset;

      if (remCount < loopCount) chunkCount = alignUp(divUp(remCount, nranks), 16/sizeof(T));

      auto modRanks = [&]__device__(int r)->int {
        return r - (r >= nranks ? nranks : 0);
      };

      // step 0: push data to next GPU
      chunk = modRanks(ringIx + nranks - 1);
      chunkOffset = chunk * chunkCount;
      offset = gridOffset + elemOffset + chunkOffset;
      nelem = (int)min(chunkCount, remCount - chunkOffset);
      prims.directSend(offset, offset, nelem);

      // k-2 steps: reduce and copy to next GPU
      for (int j = 2; j < nranks; ++j) {
        chunk = modRanks(ringIx + nranks - j);
        chunkOffset = chunk * chunkCount;
        offset = gridOffset + elemOffset + chunkOffset;
        nelem = (int)min(chunkCount, remCount - chunkOffset);
        prims.directRecvReduceDirectSend(offset, offset, nelem);
      }

      // step k-1: reduce this buffer and data, which will produce the final
      // result that we store in this data and push to the next GPU
      chunk = ringIx + 0;
      chunkOffset = chunk * chunkCount;
      offset = gridOffset + elemOffset + chunkOffset;
      nelem = (int)min(chunkCount, remCount - chunkOffset);
      prims.directRecvReduceCopyDirectSend(offset, offset, nelem, /*postOp=*/true);

      // k-2 steps: copy to next GPU
      for (int j = 1; j < nranks - 1; ++j) {
        chunk = modRanks(ringIx + nranks - j);
        chunkOffset = chunk * chunkCount;
        offset = gridOffset + elemOffset + chunkOffset;
        nelem = (int)min(chunkCount, remCount - chunkOffset);
        prims.directRecvCopyDirectSend(offset, offset, nelem);
      }

      // Make final copy from buffer to dest.
      chunk = modRanks(ringIx + 1);
      chunkOffset = chunk * chunkCount;
      offset = gridOffset + elemOffset + chunkOffset;
      nelem = (int)min(chunkCount, remCount - chunkOffset);

      prims.directRecv(offset, nelem);
    }
  }


template <typename T, typename RedOp, typename Proto>
__device__ __forceinline__ void runBineTree(int tid, int nthreads, ncclDevWorkColl* work) {
  ncclBine* bine = &ncclShmem.channel.bine;
  const int nSteps = bine->nSteps;
  const int nRanks = ncclShmem.comm.nRanks;
  const int rank   = ncclShmem.comm.rank;
 
  ssize_t gridOffset, channelCount, chunkCount;
  ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T), (ssize_t*)nullptr,
                  &gridOffset, &channelCount, &chunkCount);
  if (channelCount == 0) return;   // proxy registers nothing for empty channels
 
  const int root = ncclBineArRoot(ncclShmem.channelId, nRanks);
  int children[BINE_AR_MAX_FAN];
  int nChildren, parent;
  ncclBineArTreeNeighbors(bine->send, bine->recv, nRanks, nSteps, root, rank,
                          children, &nChildren, &parent);
 
  // Degenerate: single rank.
  if (nChildren == 0 && parent < 0) {
    const T* in = (const T*)work->sendbuff;
    T* out = (T*)work->recvbuff;
    if (in != out)
      for (ssize_t e = tid; e < channelCount; e += nthreads)
        out[gridOffset + e] = in[gridOffset + e];
    return;
  }
 
  // ---------------- root: fused reduce + broadcast ----------------
  if (parent < 0) {
    Primitives<T, RedOp, FanSymmetric<BINE_AR_MAX_FAN>, /*Direct=*/0, Proto, 0>
        prims(tid, nthreads, children, children, work->sendbuff, work->recvbuff,
              work->redOpArg, 0, 0, 0, work);
    for (ssize_t e = 0; e < channelCount; e += chunkCount) {
      const ssize_t off = gridOffset + e;
      const int nelem = (int)min(chunkCount, channelCount - e);
      prims.recvReduceCopySend(off, off, nelem, /*postOp=*/true);
    }
    return;
  }
 
  // ---------------- non-root: split threads ----------------
  int nthreadsSplit;
  if (Proto::Id == NCCL_PROTO_SIMPLE) {
    nthreadsSplit = nthreads / 2;
    if (nthreadsSplit >= 256) nthreadsSplit += 64;
  } else {  // LL / LL128: receiving from several sources costs more than sending
    nthreadsSplit = (nthreads * 7 / (10 * WARP_SIZE)) * WARP_SIZE;
  }
 
  if (tid < nthreadsSplit) {
    // Reduce up: recv from all children (+ own input), send to parent.
    Primitives<T, RedOp, FanAsymmetric<BINE_AR_MAX_FAN, 1>, /*Direct=*/0, Proto, 0>
        prims(tid, nthreadsSplit, children, &parent, work->sendbuff, work->recvbuff,
              work->redOpArg, 0 * Proto::MaxGroupWidth, 0, 0, work);
    if (nChildren == 0) {
      for (ssize_t e = 0; e < channelCount; e += chunkCount) {
        const ssize_t off = gridOffset + e;
        prims.send(off, (int)min(chunkCount, channelCount - e));
      }
    } else {
      for (ssize_t e = 0; e < channelCount; e += chunkCount) {
        const ssize_t off = gridOffset + e;
        prims.recvReduceSend(off, (int)min(chunkCount, channelCount - e));
      }
    }
  } else {
    // Broadcast down: recv from parent, write to output, forward to children.
    Primitives<T, RedOp, FanAsymmetric<1, BINE_AR_MAX_FAN>, /*Direct=*/0, Proto, 0>
        prims(tid - nthreadsSplit, nthreads - nthreadsSplit, &parent, children,
              work->sendbuff, work->recvbuff, work->redOpArg,
              1 * Proto::MaxGroupWidth, 0, 0, work);
    if (nChildren == 0) {
      for (ssize_t e = 0; e < channelCount; e += chunkCount) {
        const ssize_t off = gridOffset + e;
        prims.recv(off, (int)min(chunkCount, channelCount - e));
      }
    } else {
      for (ssize_t e = 0; e < channelCount; e += chunkCount) {
        const ssize_t off = gridOffset + e;
        prims.recvCopySend(off, (int)min(chunkCount, channelCount - e));
      }
    }
  }
}


// AllReduce = Bine RS (halving, in position space) + Bine AG (doubling), SEND buffer management.
// Position p lives at block slot p of the (recvbuff) vector. The RS leaves the fully reduced
// block at slot myIdx, and the AG starts from that same slot, so no redistribution is needed.
//
// Layout: like ring allreduce, each channel slice is cut into loops of nRanks blocks of chunkCount
// elements. Block b of loop l starts at gridOffset + l*nRanks*chunkCount + b*chunkCount.
// Tail blocks can be short or empty, but both sides derive the same size from the block index,
// and empty blocks still run a primitive call so the proxy slot count matches.

template <typename T, typename RedOp, typename Proto>
__device__ __forceinline__ void runBineSend(int tid, int nthreads,
                                            ncclDevWorkColl *work) {
  ncclBine *bine = &ncclShmem.channel.bine;
  const int steps  = bine->nDoublingSteps;   // nRanks == 1 << steps
  const int nRanks = ncclShmem.comm.nRanks;
  const int rank   = ncclShmem.comm.rank;

  ssize_t gridOffset, channelCount, chunkCount;
  ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T), (ssize_t*)nullptr,
                  &gridOffset, &channelCount, &chunkCount);
  if (channelCount == 0) return;

  // chunkSteps == 1: one block transfer must fit in exactly one FIFO step.
  // The proxy must derive the same chunk size.
  const ssize_t stepCount = (ssize_t)(Proto::calcBytePerStep() / sizeof(T));
  if (chunkCount > stepCount) chunkCount = stepCount;

  const T *sendBuf = (const T *)work->sendbuff;
  T *recvBuf = (T *)work->recvbuff;

  // ---- 0. Seed the accumulator (sendbuff stays untouched) ----
  if ((const void *)sendBuf != (void *)recvBuf) {
    for (ssize_t e = tid; e < channelCount; e += nthreads)
      recvBuf[gridOffset + e] = sendBuf[gridOffset + e];
  }
  __syncthreads();
  if (steps == 0) return;

  const int myIdx = bineSendIndex(bine->index, rank, steps);
  const bool useDirect = false; //(work->direct & (NCCL_P2P_READ | NCCL_P2P_WRITE)) ==
                         //(NCCL_P2P_READ | NCCL_P2P_WRITE);

  const ssize_t loopCount = (ssize_t)nRanks * chunkCount;
  const ssize_t nLoops    = (channelCount + loopCount - 1) / loopCount;
  const ssize_t chEnd     = gridOffset + channelCount;

  // Offset and element count of block b in loop l (clipped to the channel slice, may be 0).
  auto blk = [&](ssize_t l, int b, ssize_t &off) -> int {
    off = gridOffset + l * loopCount + (ssize_t)b * chunkCount;
    const ssize_t n = chEnd - off;
    return (int)(n <= 0 ? 0 : (n < chunkCount ? n : chunkCount));
  };

  // One butterfly step with `partner`: one Primitives, span blocks per loop, all loops.
  auto exchange = [&](int partner, int sendBeg, int recvBeg, int span,
                      bool reduce, bool doPost) {
    int peers[1] = {partner};
    auto body = [&](auto &prim, bool direct) {
      for (ssize_t l = 0; l < nLoops; ++l) {
        for (int j = 0; j < span; ++j) {
          ssize_t sOff, rOff;
          const int sN = blk(l, sendBeg + j, sOff);
          const int rN = blk(l, recvBeg + j, rOff);
          if (direct) {
            prim.directSendFromOutput(sOff, sN);
            if (reduce) prim.directRecvReduceCopy(rOff, rOff, rN, doPost);
            else        prim.directRecv(rOff, rN);
          } else {
            prim.sendFromOutput(sOff, sN);
            if (reduce) prim.recvReduceCopy(rOff, rOff, rN, doPost);
            else        prim.recv(rOff, rN);
          }
        }
      }
    };
    if (useDirect) {
      Primitives<T, RedOp, FanAsymmetric<1, 1>, 1, Proto, 0> prim(
          tid, nthreads, peers, peers, recvBuf, recvBuf, work->redOpArg, 0, 0, 0, work);
      body(prim, true);
    } else {
      Primitives<T, RedOp, FanAsymmetric<1, 1>, 0, Proto, 0> prim(
          tid, nthreads, peers, peers, recvBuf, recvBuf, work->redOpArg, 0, 0, 0, work);
      body(prim, false);
    }
  };

  // ---- 1. Reduce-scatter: halving, kernel steps steps-1 .. 0 ----
  for (int s = steps - 1; s >= 0; --s) {
    const int partner = bineSendPartner(bine->partners, rank, s, steps);
    if (partner < 0 || partner == rank) continue;   // must match the proxy's skip condition

    const int span       = 1 << s;
    const int base       = (myIdx / (span << 1)) * (span << 1);
    const bool keepLower = ((myIdx >> s) & 1) == 0;
    const int sendBeg    = keepLower ? base + span : base;
    const int recvBeg    = keepLower ? base : base + span;

    exchange(partner, sendBeg, recvBeg, span, /*reduce=*/true, /*doPost=*/(s == 0));
    __syncthreads();
  }

  // ---- 2. Allgather: doubling, kernel steps 0 .. steps-1 ----
  for (int s = 0; s < steps; ++s) {
    const int partner = bineSendPartner(bine->partners, rank, s, steps);
    if (partner < 0 || partner == rank) continue;

    const int span       = 1 << s;
    const int base       = (myIdx / (span << 1)) * (span << 1);
    const bool keepLower = ((myIdx >> s) & 1) == 0;
    const int sendBeg    = keepLower ? base : base + span;
    const int recvBeg    = keepLower ? base + span : base;

    exchange(partner, sendBeg, recvBeg, span, /*reduce=*/false, /*doPost=*/false);
    __syncthreads();
  }
}

  template<typename T, typename RedOp, typename Proto>
  __device__ __forceinline__ void runTreeUpDown(int tid, int nthreads, struct ncclDevWorkColl* work) {
    ncclTree *tree = &ncclShmem.channel.tree;
    size_t gridOffset;
    size_t channelCount;
    size_t chunkCount;
    ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T), (size_t*)nullptr, &gridOffset, &channelCount, &chunkCount);
    size_t offset;
    int nelem;

    { // Reduce : max number of recv is 3, max number of send is 1 (binary tree + local)
      Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_TREE_ARITY, 1>, /*Direct=*/1, Proto, 0> prims
        (tid, nthreads, tree->down, &tree->up, work->sendbuff, work->recvbuff, work->redOpArg, 0, 0, 0, work);
      if (tree->up == -1) {
        for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
          offset = gridOffset + elemOffset;
          nelem = min(chunkCount, channelCount - elemOffset);
          prims.directRecvReduceCopy(offset, offset, nelem, /*postOp=*/true);
        }
      }
      else if (tree->down[0] == -1) {
        for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
          offset = gridOffset + elemOffset;
          nelem = min(chunkCount, channelCount - elemOffset);
          prims.directSend(offset, offset, nelem);
        }
      }
      else {
        for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
          offset = gridOffset + elemOffset;
          nelem = min(chunkCount, channelCount - elemOffset);
          prims.directRecvReduceDirectSend(offset, offset, nelem);
        }
      }
    }

    { // Broadcast : max number of recv is 1, max number of send is 3 (binary tree + local)
      Primitives<T, RedOp, FanAsymmetric<1, NCCL_MAX_TREE_ARITY>, /*Direct=*/1, Proto, 0> prims
        (tid, nthreads, &tree->up, tree->down, work->sendbuff, work->recvbuff, work->redOpArg, 0, 0, 0, work);
      if (tree->up == -1) {
        for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
          offset = gridOffset + elemOffset;
          nelem = min(chunkCount, channelCount - elemOffset);
          prims.directSendFromOutput(offset, nelem);
        }
      }
      else if (tree->down[0] == -1) {
        for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
          offset = gridOffset + elemOffset;
          nelem = min(chunkCount, channelCount - elemOffset);
          prims.directRecv(offset, nelem);
        }
      }
      else {
        for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
          offset = gridOffset + elemOffset;
          nelem = min(chunkCount, channelCount - elemOffset);
          prims.directRecvCopyDirectSend(offset, offset, nelem);
        }
      }
    }
  }

  template<typename T, typename RedOp, typename Proto>
  __device__ __forceinline__ void runTreeSplit(int tid, int nthreads, struct ncclDevWorkColl* work) {
    ncclTree *tree = &ncclShmem.channel.tree;
    size_t gridOffset;
    size_t channelCount;
    size_t chunkCount;
    ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T), (size_t*)nullptr, &gridOffset, &channelCount, &chunkCount);
    size_t offset;
    int nelem;
    int nthreadsSplit;
    if (Proto::Id == NCCL_PROTO_SIMPLE) {
      nthreadsSplit = nthreads/2;
      if (nthreadsSplit >= 256) nthreadsSplit += 64;
    } else { // LL & LL128
      // Receiving from up to 3 sources is more compute intensive than sending
      // to 3 dests. Use 70% for reduce and 30% for bcast.
      nthreadsSplit = (nthreads*7/(10*WARP_SIZE))*WARP_SIZE;
    }

    if (tree->up == -1) {
      // Reduce and broadcast. Max number of recv is 2, max number of send is 2
      Primitives<T, RedOp, FanSymmetric<NCCL_MAX_TREE_ARITY_TOP>, /*Direct=*/1, Proto, 0>
        prims(tid, nthreads, tree->down, tree->down, work->sendbuff, work->recvbuff, work->redOpArg, 0, 0, 0, work);
      for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
        offset = gridOffset + elemOffset;
        nelem = min(chunkCount, channelCount - elemOffset);
        prims.directRecvReduceCopyDirectSend(offset, offset, nelem, /*doPost=*/true);
      }
    }
    else if (tid < nthreadsSplit) {
      /* Reduce up. Max number of recv is 3, max number of send is 1 (binary tree + local).
       * Why Direct=1????
       * Answer: Because despite not performing any direct operations, the ctor
       * must assume Direct so that it can exchange direct pointers with remote ctors
       * that are Direct, otherwise it hangs. A cleaner solution would be to seperate
       * into DirectRecv and DirectSend capabilities, this ctor would have both=0,
       * but the ctor above for tree roots would be DirectRecv=0 DirectSend=1.
       */
      // Coverity reports that the callee treats &tree->up as an array.  However, due to the use of
      // FanAsymmetric<n, 1>, only the first element is ever accessed, so it's fine.
      // coverity[callee_ptr_arith:FALSE]
      Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_TREE_ARITY, 1>, /*Direct=*/1, Proto, 0>
        prims(tid, nthreadsSplit, tree->down, &tree->up, work->sendbuff, work->recvbuff, work->redOpArg, 0*Proto::MaxGroupWidth, 0, 0, work);
      if (tree->down[0] == -1) {
        for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
          offset = gridOffset + elemOffset;
          nelem = min(chunkCount, channelCount - elemOffset);
          prims.directSend(offset, offset, nelem);
        }
      }
      else {
        for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
          offset = gridOffset + elemOffset;
          nelem = min(chunkCount, channelCount - elemOffset);
          prims.directRecvReduceDirectSend(offset, offset, nelem);
        }
      }
    }
    else {
      // Broadcast down. Max number of recv is 1, max number of send is 3 (binary tree + local)
      // Coverity reports that the callee treats &tree->up as an array.  However, due to the use of
      // FanAsymmetric<1, n>, only the first element is ever accessed, so it's fine.
      // coverity[callee_ptr_arith:FALSE]
      Primitives<T, RedOp, FanAsymmetric<1, NCCL_MAX_TREE_ARITY>, /*Direct=*/1, Proto, 0>
        prims(tid-nthreadsSplit, nthreads-nthreadsSplit, &tree->up, tree->down, work->sendbuff, work->recvbuff,
            work->redOpArg, 1*Proto::MaxGroupWidth, 0, 0, work);
      if (tree->down[0] == -1) {
        for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
          offset = gridOffset + elemOffset;
          nelem = min(chunkCount, channelCount - elemOffset);
          prims.directRecv(offset, nelem);
        }
      }
      else {
        for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
          offset = gridOffset + elemOffset;
          nelem = min(chunkCount, channelCount - elemOffset);
          prims.directRecvCopyDirectSend(offset, offset, nelem);
        }
      }
    }
  }
}

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    using Proto = ProtoSimple<ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS>;
    runRing<T, RedOp, Proto>(tid, nthreads, work);
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    #if CUDART_VERSION >= 11020 && CUDART_VERSION < 11040 && __CUDA_ARCH__ >= 800
      runTreeUpDown<T, RedOp, ProtoSimple<1, 1>>(tid, nthreads, work);
    #else
      runTreeSplit<T, RedOp, ProtoSimple<1, 1>>(tid, nthreads, work);
    #endif
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_COLLNET_DIRECT, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int/*nthreads*/, struct ncclDevWorkColl* work) {
    static constexpr int COLLNET_COPY_THREADS = 96;
    const int bid = ncclShmem.channelId - work->channelLo;
    const int nChannels = work->channelHi - work->channelLo + 1;
    struct ncclDirect* direct = &ncclShmem.channel.collnetDirect;
    const ssize_t chunkSize = work->collnet.chunkCount;
    const ssize_t size = work->collnet.count;
    const ssize_t loopSize = nChannels*direct->nHeads*chunkSize;

    const int hasUp = (direct->up[0] >= 0) ? 1 : 0;
    const int hasDn = (direct->down[0] >= 0) ? 1 : 0;
    const int nThreadsScatter = WARP_SIZE + ((hasUp && hasDn) ? COLLNET_COPY_THREADS : hasUp ? 3*COLLNET_COPY_THREADS : 0);
    const int nThreadsGather  =             ((hasUp && hasDn) ? COLLNET_COPY_THREADS : hasUp ? 2*COLLNET_COPY_THREADS : 0);
    const int nThreadsBcast   = WARP_SIZE + ((hasUp && hasDn) ? COLLNET_COPY_THREADS : hasUp ? 0 : 2*COLLNET_COPY_THREADS);
    const int nThreadsReduce = work->nWarps*WARP_SIZE - nThreadsScatter - nThreadsGather - nThreadsBcast;
    const int tidStartBcast = nThreadsGather;
    const int tidStartScatter = tidStartBcast + nThreadsBcast;
    const int tidStartReduce = tidStartScatter + nThreadsScatter;
    using Proto = ProtoSimple<1, 1>;

    if (tid >= tidStartScatter && tid < tidStartReduce && hasUp) {
      // Scatter
      Primitives<T, RedOp, FanAsymmetric<0, NCCL_MAX_DIRECT_ARITY>, /*Direct=*/0, Proto, 0>
        prims(tid-tidStartScatter, nThreadsScatter, NULL, direct->up, work->sendbuff, work->recvbuff,
           work->redOpArg, 2*Proto::MaxGroupWidth, 1, 1, work);
      ssize_t offsetBase, peerOffset;
      ssize_t maxNelems;
      if (work->netRegUsed) {
        offsetBase = bid * chunkSize;
        maxNelems = size;  // never be the min
        peerOffset = nChannels * chunkSize;
      } else {
        offsetBase = bid * direct->nHeads * chunkSize;
        maxNelems = direct->nHeads * chunkSize;
        peerOffset = chunkSize;
      }
      // For collnet UB case, we need to organize buffers differently for contiguous buffer access
      // across channels. This access pattern should be consistent with code in coll_net.cc
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        ssize_t offset = gridOffset + offsetBase;
        ssize_t nelem = min(maxNelems, size - offset);
        prims.scatter(offset, nelem, chunkSize, peerOffset, direct->headRank, direct->shift);
      }
      // Coverity complains about a possible overrun inside the destructor of "prims", but that's actually
      // a false positive.
      // coverity[overrun-call:FALSE]
    } else if (tid >= tidStartReduce && direct->out != -1) {
      if (hasDn) {
        // Reduce, send to network
        Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_DIRECT_ARITY, 1>, /*Direct=*/0, Proto, 0>
          prims(tid-tidStartReduce, nThreadsReduce, direct->down, &direct->out, work->sendbuff, work->recvbuff,
             work->redOpArg, 3*Proto::MaxGroupWidth, 1, 1, work);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = work->netRegUsed ? gridOffset + (bid + direct->headRank * nChannels) * chunkSize
                                    : gridOffset + (bid * direct->nHeads + direct->headRank) * chunkSize;
          int nelem = min(chunkSize, size - offset);
          prims.recvReduceDirectSend(offset, offset, nelem);
        }
      } else {
        // Directly send to network
        if (work->netRegUsed) {
          if (tid == tidStartReduce) {
            Primitives<T, RedOp, FanAsymmetric<0, 1>, /*Direct=*/0, Proto, 0>::sendPeerNotify(direct->out, 1, 1);
          }
          __syncwarp();
        } else {
          Primitives<T, RedOp, FanAsymmetric<0, 1>, /*Direct=*/0, Proto, 0>
          prims(tid-tidStartReduce, nThreadsReduce, nullptr, &direct->out, work->sendbuff, work->recvbuff,
             work->redOpArg, 3*Proto::MaxGroupWidth, 1, 1);
          for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
            ssize_t offset = gridOffset + (bid * direct->nHeads + direct->headRank) * chunkSize;
            int nelem = min(chunkSize, size - offset);
            prims.send(offset, nelem);
          }
        }
      }
    } else if (tid < tidStartBcast && hasUp) {
      // Gather
      Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_DIRECT_ARITY, 0>, /*Direct=*/0, Proto, 0>
        prims(tid, nThreadsGather, direct->up, NULL, work->sendbuff, work->recvbuff,
           work->redOpArg, 0*Proto::MaxGroupWidth, 0, 0, work);
      ssize_t offsetBase, peerOffset;
      ssize_t maxNelems;
      if (work->netRegUsed) {
        offsetBase = bid * chunkSize;
        maxNelems = size;  // never be the min
        peerOffset = nChannels * chunkSize;
      } else {
        offsetBase = bid * direct->nHeads * chunkSize;
        maxNelems = direct->nHeads * chunkSize;
        peerOffset = chunkSize;
      }
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        ssize_t offset = gridOffset + offsetBase;
        ssize_t nelem = min(maxNelems, size - offset);
        prims.directGather(offset, nelem, chunkSize, peerOffset, direct->headRank, direct->shift);
      }
    } else if (tid >= tidStartBcast && tid < tidStartScatter && direct->out != -1) {
      if (hasDn) {
        // Recv from network, broadcast
        // Coverity complains about a possible overrun inside the class below, but that's actually
        // a false positive.
        // coverity[identity_transfer:FALSE]
        Primitives<T, RedOp, FanAsymmetric<1, NCCL_MAX_DIRECT_ARITY>, /*Direct=*/0, Proto, 0>
          prims(tid-tidStartBcast, nThreadsBcast, &direct->out, direct->down, work->sendbuff, work->recvbuff,
             work->redOpArg, 1*Proto::MaxGroupWidth, 0, 0, work);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = work->netRegUsed ? gridOffset + (bid + direct->headRank * nChannels) * chunkSize
                                            : gridOffset + (bid * direct->nHeads + direct->headRank) * chunkSize;
          int nelem = min(chunkSize, size - offset);
          prims.directRecvCopyDirectSend(offset, offset, nelem, /*postOp=*/true);
        }
      } else {
        if (work->netRegUsed) {
          if (tid == tidStartBcast) {
            Primitives<T, RedOp, FanAsymmetric<1, 0>, /*Direct=*/0, Proto, 0>::recvPeerNotify(direct->out, 0, 1);
          }
          __syncwarp();
        } else {
          // Recv from network (no post thread needed)
          Primitives<T, RedOp, FanAsymmetric<1, 0>, /*Direct=*/0, Proto, 0>
            prims(tid - tidStartBcast, nThreadsBcast, &direct->out, nullptr, work->sendbuff, work->recvbuff,
              work->redOpArg, 1 * Proto::MaxGroupWidth, 0, 0);
          for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
            ssize_t offset = gridOffset + (bid * direct->nHeads + direct->headRank) * chunkSize;
            int nelem = min(chunkSize, size - offset);
            prims.recv(offset, nelem, /*postOp=*/true);
          }
        }
      }
    }
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_NVLS, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int/*nthreads*/, struct ncclDevWorkColl* work) {
    struct ncclNvls* nvls = &ncclShmem.channel.nvls;
    const bool hasOut = nvls->out != -1;
    const int nranks = ncclShmem.comm.nRanks;
    const int totalWarps = NCCL_MAX_NTHREADS/WARP_SIZE;
    const int bcastWarps = hasOut ? (work->regUsed ? ((totalWarps - 2) >> 1) - 1 : 2) : 0;
    const int reduceWarps = work->regUsed ? (totalWarps - bcastWarps - 2) : (hasOut ? 3 : nranks <= 6 ? 7 : 5);
    const int scatterWarps = work->regUsed ? 1 : (totalWarps - reduceWarps - bcastWarps + 1) >> 1;
    const int gatherWarps = work->regUsed ? 1 : (totalWarps - reduceWarps - bcastWarps) >> 1;

    const int nThreadsScatter = scatterWarps*WARP_SIZE;
    const int nThreadsGather  = gatherWarps*WARP_SIZE;
    const int nThreadsReduce = reduceWarps*WARP_SIZE;
    const int nThreadsBcast  = (bcastWarps)*WARP_SIZE;
    const int tidEndScatter = nThreadsScatter;
    const int tidEndGather = tidEndScatter + nThreadsGather;
    const int tidEndReduce = tidEndGather + nThreadsReduce;
    const int tidEndBcast = tidEndReduce + nThreadsBcast;

    if (work->oneNode) {
      ssize_t gridOffset, channelCount, chunkSize;
      ncclCollCbdPart(work, ncclShmem.channelId, NCCL_PROTO_SIMPLE, sizeof(T), (ssize_t*)nullptr, &gridOffset, &channelCount, &chunkSize);
      const ssize_t loopCount = nvls->nHeads * chunkSize;
      int remCount = channelCount%(nvls->nHeads*chunkSize);
      int lastChunkSize = alignUp(divUp(remCount, nvls->nHeads), 16384/sizeof(T));

      if (tid < tidEndScatter) {
        // Scatter
        using Proto = ProtoSimple<1, 1, COLL_UNROLL>;
        Primitives<T, RedOp, FanAsymmetric<0, NCCL_MAX_NVLS_ARITY>, /*Direct=*/0, Proto, 0>
          prims(tid, nThreadsScatter, NULL, nvls->up, work->sendbuff, NULL,
            work->redOpArg, 0 * Proto::MaxGroupWidth, 1, 1);
        for (ssize_t elemOffset = 0; elemOffset < channelCount; elemOffset += loopCount) {
          if (channelCount - elemOffset < loopCount) chunkSize = lastChunkSize;
          ssize_t offset = gridOffset + elemOffset;
          int nelem = work->regUsed ? 0 : min(loopCount, channelCount - elemOffset);
          prims.scatter(offset, nelem, chunkSize, chunkSize, -1, 0);
        }
      } else if (tid < tidEndGather) {
        // Gather
        using Proto = ProtoSimple<1, 1, COLL_UNROLL>;
        Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_NVLS_ARITY, 0>, /*Direct=*/0, Proto, 0>
          prims(tid - tidEndScatter, nThreadsGather, nvls->up, NULL, NULL, work->recvbuff,
            work->redOpArg, 1 * Proto::MaxGroupWidth, 1, 1);
        for (ssize_t elemOffset = 0; elemOffset < channelCount; elemOffset += loopCount) {
          if (channelCount - elemOffset < loopCount) chunkSize = lastChunkSize;
          ssize_t offset = gridOffset + elemOffset;
          int nelem = work->regUsed ? 0 : min(loopCount, channelCount - elemOffset);
          prims.gather(offset, nelem, chunkSize, chunkSize, -1, 0);
        }
      } else if (tid < tidEndReduce && nvls->headRank != -1) {
        // Reduce, broadcast through NVLS
        using Proto = ProtoSimple<1, 1, COLL_UNROLL, 1, 1>;
        Primitives<T, RedOp, FanSymmetric<1>, /*Direct=*/1, Proto, 0>
          prims(tid - tidEndGather, nThreadsReduce, &nvls->down, &nvls->down, NULL, NULL,
            work->redOpArg, 2 * Proto::MaxGroupWidth, 0, 0, work);
        for (ssize_t elemOffset = 0; elemOffset < channelCount; elemOffset += loopCount) {
          ssize_t chunkOffset, offset;
          int nelem;
          if (channelCount - elemOffset < loopCount) chunkSize = lastChunkSize;
          chunkOffset = elemOffset + nvls->headRank * chunkSize;
          offset = gridOffset + chunkOffset;
          nelem = min(chunkSize, channelCount - chunkOffset);
          prims.directRecvDirectSend(offset, offset, nelem);
        }
      }
    } else {
      const int bid = ncclShmem.channelId - work->channelLo;
      const int nChannels = work->channelHi - work->channelLo + 1;
      const ssize_t chunkSize = work->collnet.chunkCount;
      const ssize_t loopSize = nChannels * nvls->nHeads * chunkSize;
      const ssize_t size = work->collnet.count;

      if (tid < tidEndScatter) {
        // Scatter
        using Proto = ProtoSimple<1, 1, COLL_UNROLL>;
        Primitives<T, RedOp, FanAsymmetric<0, NCCL_MAX_NVLS_ARITY>, /*Direct=*/0, Proto, 0>
          prims(tid, nThreadsScatter, NULL, nvls->up, work->sendbuff, NULL,
            work->redOpArg, 0 * Proto::MaxGroupWidth, 1, 1);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid * nvls->nHeads * chunkSize;
          int nelem = work->regUsed ? 0 : min(nvls->nHeads * chunkSize, size - offset);
          prims.scatter(offset, nelem, chunkSize, chunkSize, -1, 0);
        }
        // coverity[overrun-call] => Coverity think prims.index can be greater than 1
      } else if (tid < tidEndGather) {
        // Gather
        using Proto = ProtoSimple<1, 1, COLL_UNROLL>;
        Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_NVLS_ARITY, 0>, /*Direct=*/0, Proto, 0>
          prims(tid - tidEndScatter, nThreadsGather, nvls->up, NULL, NULL, work->recvbuff,
            work->redOpArg, 1 * Proto::MaxGroupWidth, 1, 1);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid * nvls->nHeads * chunkSize;
          int nelem = work->regUsed ? 0 : min(nvls->nHeads * chunkSize, size - offset);
          prims.gather(offset, nelem, chunkSize, chunkSize, -1, 0);
        }
      } else if (tid < tidEndReduce && nvls->headRank != -1) {
        // Reduce, send to network
        using Proto = ProtoSimple<1, 1, COLL_UNROLL, 1, 0>;
        // Coverity complains about a possible overrun inside the class below, but that's actually
        // a false positive.
        // coverity[identity_transfer:FALSE]
        Primitives<T, RedOp, FanSymmetric<1>, /*Direct=*/1, Proto, 0>
        prims(tid - tidEndGather, nThreadsReduce, &nvls->down, &nvls->out, NULL, work->recvbuff,
          work->redOpArg, 2 * Proto::MaxGroupWidth, 0, 1, work);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = work->regUsed && work->netRegUsed ? gridOffset + (nvls->headRank * nChannels + bid) * chunkSize
                                                             : gridOffset + (bid * nvls->nHeads + nvls->headRank) * chunkSize;
          int nelem = min(chunkSize, size - offset);
          prims.directRecvDirectSend(offset, offset, nelem);
        }
      } else if (tid < tidEndBcast && nvls->headRank != -1) {
        // Recv from network, broadcast
        using Proto = ProtoSimple<1, 1, COLL_UNROLL, 0, 1>;
        // Coverity complains about a possible overrun inside the class below, but that's actually
        // a false positive.
        // coverity[identity_transfer:FALSE]
        Primitives<T, RedOp, FanSymmetric<1>, /*Direct=*/1, Proto, 0>
          prims(tid - tidEndReduce, nThreadsBcast, &nvls->out, &nvls->down, NULL, work->recvbuff,
            work->redOpArg, 3 * Proto::MaxGroupWidth, 0, 0, work);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = work->regUsed && work->netRegUsed ? gridOffset + (nvls->headRank * nChannels + bid) * chunkSize
                                                             : gridOffset + (bid * nvls->nHeads + nvls->headRank) * chunkSize;
          int nelem = min(chunkSize, size - offset);
          prims.directRecvDirectSend(offset, offset, nelem);
        }
      }
    }
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_NVLS_TREE, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int/*nthreads*/, struct ncclDevWorkColl* work) {
    struct ncclNvls* nvls = &ncclShmem.channel.nvls;
    const int treeUp = nvls->treeUp;
    const int* treeDown = nvls->treeDown;
    ssize_t gridOffset, channelCount, chunkCount;
    ncclCollCbdPart(work, ncclShmem.channelId, NCCL_PROTO_SIMPLE, sizeof(T), (ssize_t*)nullptr, &gridOffset, &channelCount, &chunkCount);
    const ssize_t loopCount = nvls->nHeads * chunkCount;
    const int nranks = ncclShmem.comm.nRanks;
    const bool hasUp = treeUp != -1;
    const int totalWarps = NCCL_MAX_NTHREADS/WARP_SIZE;
    const int bcastWarps = hasUp ? (work->regUsed ? ((totalWarps - 2) >> 1) - 1 : 4) : 0;
    const int reduceWarps = work->regUsed ? (totalWarps - bcastWarps - 2) : (hasUp ? 5 : nranks <= 6 ? 7 : 5);
    const int scatterWarps = work->regUsed ? 1 : (totalWarps - reduceWarps - bcastWarps + 1) >> 1;
    const int gatherWarps = work->regUsed ? 1 : (totalWarps - reduceWarps - bcastWarps) >> 1;
    ssize_t offset;
    int nelem;
    int remCount = channelCount%(nvls->nHeads*chunkCount);
    int lastChunkCount = alignUp(divUp(remCount, nvls->nHeads), 16/sizeof(T));

    const int nThreadsScatter = scatterWarps*WARP_SIZE;
    const int nThreadsGather  = gatherWarps*WARP_SIZE;
    const int nThreadsReduce = reduceWarps*WARP_SIZE;
    const int nThreadsBcast  = (bcastWarps)*WARP_SIZE;
    const int tidEndScatter = nThreadsScatter;
    const int tidEndGather = tidEndScatter + nThreadsGather;
    const int tidEndReduce = tidEndGather + nThreadsReduce;
    const int tidEndBcast = tidEndReduce + nThreadsBcast;

    if (tid < tidEndScatter) {
      // Scatter
      using Proto = ProtoSimple<1, 1, COLL_UNROLL>;
      Primitives<T, RedOp, FanAsymmetric<0, NCCL_MAX_NVLS_ARITY>, /*Direct=*/0, Proto, 0>
        prims(tid, nThreadsScatter, NULL, nvls->up, work->sendbuff, NULL,
          work->redOpArg, 0 * Proto::MaxGroupWidth, 1, 1);
      for (ssize_t elemOffset = 0; elemOffset < channelCount; elemOffset += loopCount) {
        if (channelCount - elemOffset < loopCount) chunkCount = lastChunkCount;
        offset = gridOffset + elemOffset;
        nelem = work->regUsed ? 0 : min(loopCount, channelCount - elemOffset);
        prims.scatter(offset, nelem, chunkCount, chunkCount, -1, 0);
      }
    } else if (tid < tidEndGather) {
      // Gather
      using Proto = ProtoSimple<1, 1, COLL_UNROLL>;
      Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_NVLS_ARITY, 0>, /*Direct=*/0, Proto, 0>
        prims(tid - tidEndScatter, nThreadsGather, nvls->up, NULL, NULL, work->recvbuff,
          work->redOpArg, 1 * Proto::MaxGroupWidth, 1, 1);
      for (ssize_t elemOffset = 0; elemOffset < channelCount; elemOffset += loopCount) {
        if (channelCount - elemOffset < loopCount) chunkCount = lastChunkCount;
        offset = gridOffset + elemOffset;
        nelem = work->regUsed ? 0 : min(loopCount, channelCount - elemOffset);
        prims.gather(offset, nelem, chunkCount, chunkCount, -1, 0);
      }
    } else if (tid < tidEndReduce && nvls->headRank != -1) {
      if (!hasUp) {
        // Reduce and Broadcast
        using Proto = ProtoSimple<1, 1, COLL_UNROLL, 1, 1>;
        Primitives<T, RedOp, FanSymmetric<3>, /*Direct=*/1, Proto, 0>
          prims(tid - tidEndGather, nThreadsReduce, treeDown, treeDown, NULL, NULL,
            work->redOpArg, 2 * Proto::MaxGroupWidth, 0, 0, work);
        for (ssize_t elemOffset = 0; elemOffset < channelCount; elemOffset += loopCount) {
          ssize_t chunkOffset;
          if (channelCount - elemOffset < loopCount) chunkCount = lastChunkCount;
          chunkOffset = elemOffset + nvls->headRank * chunkCount;
          offset = gridOffset + chunkOffset;
          nelem = min(chunkCount, channelCount - chunkOffset);
          prims.directRecvDirectSend(offset, offset, nelem);
        }
      } else {
        // Reduce, send to network
        using Proto = ProtoSimple<1, 1, COLL_UNROLL, 1, 0>;
        // Coverity reports that the callee treats &treeUp as an array.  However, due to the use of
        // FanAsymmetric<3, 1>, only the first element is ever accessed, so it's fine.
        // coverity[callee_ptr_arith:FALSE]
        Primitives<T, RedOp, FanAsymmetric<3, 1>, /*Direct=*/1, Proto, 0>
          prims(tid - tidEndGather, nThreadsReduce, treeDown, &treeUp, NULL, NULL,
            work->redOpArg, 2 * Proto::MaxGroupWidth, 0, 0, work);
        for (ssize_t elemOffset = 0; elemOffset < channelCount; elemOffset += loopCount) {
          ssize_t chunkOffset;
          if (channelCount - elemOffset < loopCount) chunkCount = lastChunkCount;
          chunkOffset = elemOffset + nvls->headRank * chunkCount;
          offset = gridOffset + chunkOffset;
          nelem = min(chunkCount, channelCount - chunkOffset);
          prims.directRecvDirectSend(offset, offset, nelem);
        }
      }
    } else if (tid < tidEndBcast && nvls->headRank != -1) {
      // Recv from network, broadcast
      using Proto = ProtoSimple<1, 1, COLL_UNROLL, 0, 1>;
      // Coverity reports that the callee treats &treeUp as an array.  However, due to the use of
      // FanAsymmetric<1, 3>, only the first element is ever accessed, so it's fine.
      // coverity[callee_ptr_arith:FALSE]
      Primitives<T, RedOp, FanAsymmetric<1, 3>, /*Direct=*/1, Proto, 0>
        prims(tid - tidEndReduce, nThreadsBcast, &treeUp, treeDown, NULL, NULL,
          work->redOpArg, 3 * Proto::MaxGroupWidth, 0, 0, work);
      for (ssize_t elemOffset = 0; elemOffset < channelCount; elemOffset += loopCount) {
        ssize_t chunkOffset;
        if (channelCount - elemOffset < loopCount) chunkCount = lastChunkCount;
        chunkOffset = elemOffset + nvls->headRank * chunkCount;
        offset = gridOffset + chunkOffset;
        nelem = min(chunkCount, channelCount - chunkOffset);
        prims.directRecvDirectSend(offset, offset, nelem);
      }
    }
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_COLLNET_CHAIN, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    const int bid = ncclShmem.channelId - work->channelLo;
    const int nChannels = work->channelHi - work->channelLo + 1;
    ncclTree *tree = &ncclShmem.channel.collnetChain;
    ssize_t chunkSize = work->collnet.chunkCount;
    const ssize_t loopSize = int(nChannels*chunkSize);
    const int nranks = ncclShmem.comm.nRanks;
    const ssize_t size = work->collnet.count;

    int nthreadsSplit = nthreads/2;
    if (nthreadsSplit >= 256) nthreadsSplit += 64;

    int group, connIndex, send, recv, groupTid, groupNthreads;
    using Proto = ProtoSimple<1, 1>;
    if (tid < nthreadsSplit) {
      // Reduce up the chain
      group = 0;
      connIndex = 1;
      recv = tree->down[0];
      send = tree->up;
      groupTid = tid;
      groupNthreads = nthreadsSplit;
    } else {
      // Broadcast down the chain
      group = 1;
      connIndex = 0;
      recv = tree->up;
      send = tree->down[0];
      groupTid = tid - nthreadsSplit;
      groupNthreads = nthreads-nthreadsSplit;
    }

    if (tid < nthreadsSplit) {
      if (recv == -1) {
        if (work->netRegUsed) {
          if (groupTid == 0) {
            Primitives<T, RedOp, FanSymmetric<1>, /*Direct=*/1, Proto, 0>::sendPeerNotify(send, connIndex, 1);
          }
          __syncwarp();
        } else {
          Primitives<T, RedOp, FanSymmetric<1>, /*Direct=*/1, Proto, 0>
            prims(groupTid, groupNthreads, &recv, &send, work->sendbuff, work->recvbuff,
              work->redOpArg, group * Proto::MaxGroupWidth, connIndex, connIndex, work);
          for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
            ssize_t offset = gridOffset + bid * int(chunkSize);
            int nelem = min(chunkSize, size - offset);
            // coverity[overrun-call] => Coverity think prims.index can be greater than 1
            prims.directSend(offset, offset, nelem);
          }
          // coverity[overrun-call] => Coverity think prims.index can be greater than 1
        }
      } else {
        Primitives<T, RedOp, FanSymmetric<1>, /*Direct=*/1, Proto, 0>
          prims(groupTid, groupNthreads, &recv, &send, work->sendbuff, work->recvbuff,
            work->redOpArg, group * Proto::MaxGroupWidth, connIndex, connIndex, work);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid * int(chunkSize);
          int nelem = min(chunkSize, size - offset);
          // coverity[overrun-call] => Coverity think prims.index can be greater than 1
          prims.directRecvReduceDirectSend(offset, offset, nelem);
        }
        // coverity[overrun-call] => Coverity think prims.index can be greater than 1
      }
    }
    else {
      if (recv == nranks) {
        // I'm the first in the broadcast chain, I need to perform the division (postOp)
        if (send == -1) {
          if (work->netRegUsed) {
            if (groupTid == 0) {
              Primitives<T, RedOp, FanSymmetric<1>, /*Direct=*/1, Proto, 0>::recvPeerNotify(recv, connIndex, 1);
            }
            __syncwarp();
          } else {
            // Coverity reports that the callee treats &send as an array.  However, due to the use of
            // FanSymmetric<1>, only the first element is ever accessed, so it's fine.
            // coverity[callee_ptr_arith:FALSE]
            Primitives<T, RedOp, FanSymmetric<1>, /*Direct=*/1, Proto, 0>
              prims(groupTid, groupNthreads, &recv, &send, work->sendbuff, work->recvbuff,
                work->redOpArg, group * Proto::MaxGroupWidth, connIndex, connIndex, work);
            for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
              ssize_t offset = gridOffset + bid * int(chunkSize);
              int nelem = min(chunkSize, size - offset);
              prims.directRecv(offset, nelem, /*postOp*/true);
            }
          }
        } else {
          // Coverity reports that the callee treats &send as an array.  However, due to the use of
          // FanSymmetric<1>, only the first element is ever accessed, so it's fine.
          // coverity[callee_ptr_arith:FALSE]
          Primitives<T, RedOp, FanSymmetric<1>, /*Direct=*/1, Proto, 0>
            prims(groupTid, groupNthreads, &recv, &send, work->sendbuff, work->recvbuff,
              work->redOpArg, group * Proto::MaxGroupWidth, connIndex, connIndex, work);
          for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
            ssize_t offset = gridOffset + bid * int(chunkSize);
            int nelem = min(chunkSize, size - offset);
            prims.directRecvCopyDirectSend(offset, offset, nelem, /*postOp*/true);
          }
        }
      } else {
        // Coverity reports that the callee treats &send as an array.  However, due to the use of
        // FanSymmetric<1>, only the first element is ever accessed, so it's fine.
        // coverity[callee_ptr_arith:FALSE]
        Primitives<T, RedOp, FanSymmetric<1>, /*Direct=*/1, Proto, 0>
          prims(groupTid, groupNthreads, &recv, &send, work->sendbuff, work->recvbuff,
            work->redOpArg, group * Proto::MaxGroupWidth, connIndex, connIndex, work);
        if (send == -1) {
          for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
            ssize_t offset = gridOffset + bid*int(chunkSize);
            int nelem = min(chunkSize, size-offset);
            prims.directRecv(offset, nelem);
          }
        } else {
          for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
            ssize_t offset = gridOffset + bid*int(chunkSize);
            int nelem = min(chunkSize, size-offset);
            prims.directRecvCopyDirectSend(offset, offset, nelem);
          }
        }
      }
    }
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    runRing<T, RedOp, ProtoLL>(tid, nthreads, work);
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_TREE, NCCL_PROTO_LL> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    runTreeSplit<T, RedOp, ProtoLL>(tid, nthreads, work);
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL128> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    runRing<T, RedOp, ProtoLL128>(tid, nthreads, work);
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_TREE, NCCL_PROTO_LL128> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    runTreeSplit<T, RedOp, ProtoLL128>(tid, nthreads, work);
  }
};


template <typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_BINE, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl *work) {
    ncclBine *bine = &ncclShmem.channel.bine;

    // Same chunkSteps=1 hack as the RS/AG specializations.
    using Proto = ProtoSimple<1, 1>;

    if (bine->bufferManagement == SEND)
      runBineSend<T, RedOp, Proto>(tid, nthreads, work);
    else if (bine->bufferManagement == TREE)
      runBineTree<T, RedOp, Proto>(tid, nthreads, work);
    else {
      // Connect/SaveProxyBine only register the tree phases for non-SEND, so there is no
      // doubling connectivity here.
      assert(false && "Bine AllReduce doubling path requires SEND buffer management");
    }
  }
};
