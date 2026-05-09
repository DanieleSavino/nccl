/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "device.h"
#include "collectives.h"
#include "primitives.h"

namespace {
  template<typename T, typename RedOp, typename Proto>
  __device__ __forceinline__ void runRing(int tid, int nthreads, struct ncclDevWorkColl* work) {
    ncclRing *ring = &ncclShmem.channel.ring;
    const int rank = ring->userRanks[0];
    const int nextRank = ring->userRanks[1];
    const int root = work->root;
    ssize_t chunkCount;
    ssize_t channelCount;
    ssize_t gridOffset;
    ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T), (ssize_t*)nullptr, &gridOffset, &channelCount, &chunkCount);
    size_t offset;
    int nelem;
    int workNthreads;
    bool isNetOffload = work->isOneRPN && work->netRegUsed;

    T *inputBuf = (T*)work->sendbuff;
    T *outputBuf = (T*)work->recvbuff;
    workNthreads = isNetOffload ? WARP_SIZE : nthreads;

    if (tid < workNthreads) {
      // Coverity reports that the callee treats &ring->next as an array.  However, due to the use of
      // FanSymmetric<1>, only the first element is ever accessed, so it's fine.
      // coverity[callee_ptr_arith:FALSE]
      Primitives<T, RedOp, FanSymmetric<1>, 1, Proto, 0>
        prims(tid, workNthreads, &ring->prev, &ring->next, inputBuf, outputBuf, work->redOpArg, 0, 0, 0, work);

      for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
        offset = gridOffset + elemOffset;
        nelem = min(chunkCount, channelCount - elemOffset);

        if (rank == root) {
          if (inputBuf == outputBuf || isNetOffload) {
            prims.directSend(offset, offset, nelem);
          } else {
            prims.directCopySend(offset, offset, nelem);
          }
        } else if (nextRank == root) {
          prims.directRecv(offset, nelem);
        } else {
          prims.directRecvCopyDirectSend(offset, offset, nelem);
        }
      }
    } else if (inputBuf != outputBuf && rank == root) {
      inputBuf = inputBuf + gridOffset;
      outputBuf = outputBuf + gridOffset;
      reduceCopy<COLL_UNROLL, RedOp, T, 0, 1, 1, 0, 1, 1, /*PreOpSrcs=*/0>
        (tid - workNthreads, nthreads - workNthreads, work->redOpArg, false, 1, (void**)&inputBuf, 1, (void**)&outputBuf, channelCount);
    }
    if (isNetOffload) barrier_sync(14, nthreads);
  }
}

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncBroadcast, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    using Proto = ProtoSimple<BROADCAST_CHUNKSTEPS/BROADCAST_SLICESTEPS, BROADCAST_SLICESTEPS>;
    runRing<T, RedOp, Proto>(tid, nthreads, work);
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncBroadcast, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    runRing<T, RedOp, ProtoLL>(tid, nthreads, work);
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncBroadcast, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL128> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    runRing<T, RedOp, ProtoLL128>(tid, nthreads, work);
  }
};

// FIXME: [HLC] Move this in proper helper
__device__ __forceinline__ int bine_rank2nb(int rank, int s) {
  const int size = (1 << s);
  const uint32_t mask = 0xAAAAAAAAu;
  const int val = (int)((mask + (uint32_t)rank) ^ mask);
  return val & (size - 1);
}

__device__ __forceinline__ int bine_nb2rank(int nb, int s) {
  const int size = (1 << s);
  const uint32_t mask = 0xAAAAAAAAu;
  const int val = (int)((mask ^ (uint32_t)nb) - mask);
  return val & (size - 1);
}

// INFO: [HLC] Added bine bcast
template <typename T, typename RedOp, typename Proto>
__device__ __forceinline__ void runBine(int tid, int nthreads,
                                        struct ncclDevWorkColl *work) {
  const int rank = ncclShmem.comm.rank;
  const int nranks = ncclShmem.comm.nRanks;
  const int root = work->root;
  const int s = __log2f(nranks);

  ssize_t chunkCount, channelCount, gridOffset;
  ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T),
                  (ssize_t *)nullptr, &gridOffset, &channelCount, &chunkCount);

  T *inputBuf = (T *)work->sendbuff;
  T *outputBuf = (T *)work->recvbuff;

  // Root copies inputBuf -> outputBuf if they differ
  if (rank == root && inputBuf != outputBuf) {
    for (ssize_t i = tid; i < channelCount; i += nthreads)
      outputBuf[gridOffset + i] = inputBuf[gridOffset + i];
    __syncthreads();
  }

  int mod_rank = (rank - root + nranks) % nranks;
  int nb_rank = bine_rank2nb(mod_rank, s);
  int recvd = (rank == root) ? 1 : 0;

  if (tid == 0) printf("[BINE bcast] rank=%d nranks=%d root=%d\n", rank, nranks, root);

  // FIXME: [HLC] Create an actual bine communicator
  int mask = 1 << (s - 1);
  while (mask > 0) {
    int mask_lsbs = (mask << 1) - 1;
    int nb_peer = nb_rank ^ mask_lsbs;
    int mod_peer = bine_nb2rank(nb_peer, s);
    int peer = (mod_peer + root) % nranks;

    int do_send = 0, do_recv = 0;
    if (recvd) {
      do_send = 1;
    } else {
      int eq_lsbs = nb_rank & mask_lsbs;
      if (eq_lsbs == 0 || eq_lsbs == mask_lsbs) {
        do_recv = 1;
      }
    }

    if (do_send) {
      // Always send from outputBuf: root copied there above, non-root received there
      Primitives<T, RedOp, FanAsymmetric<0, 1>, 1, Proto, 0> prims(
          tid, nthreads, nullptr, &peer, outputBuf, outputBuf,
          work->redOpArg, 0, 0, 0, work);
      for (ssize_t elemOffset = 0; elemOffset < channelCount;
           elemOffset += chunkCount) {
        ssize_t offset = gridOffset + elemOffset;
        int nelem = (int)min(chunkCount, channelCount - elemOffset);
        prims.send(offset, nelem);
      }
    } else if (do_recv) {
      Primitives<T, RedOp, FanAsymmetric<1, 0>, 1, Proto, 0> prims(
          tid, nthreads, &peer, nullptr, inputBuf, outputBuf,
          work->redOpArg, 0, 0, 0, work);
      for (ssize_t elemOffset = 0; elemOffset < channelCount;
           elemOffset += chunkCount) {
        ssize_t offset = gridOffset + elemOffset;
        int nelem = (int)min(chunkCount, channelCount - elemOffset);
        prims.recv(offset, nelem);
      }
      recvd = 1;
    }

    mask >>= 1;
  }
}

template <typename T, typename RedOp>
struct RunWorkColl<ncclFuncBroadcast, T, RedOp, NCCL_ALGO_BINE,
                   NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int nthreads,
                                      struct ncclDevWorkColl *work) {
    using Proto = ProtoSimple<BROADCAST_CHUNKSTEPS / BROADCAST_SLICESTEPS,
                              BROADCAST_SLICESTEPS>;
    runBine<T, RedOp, Proto>(tid, nthreads, work);
  }
};

template <typename T, typename RedOp>
struct RunWorkColl<ncclFuncBroadcast, T, RedOp, NCCL_ALGO_BINE, NCCL_PROTO_LL> {
  __device__ __forceinline__ void run(int tid, int nthreads,
                                      struct ncclDevWorkColl *work) {
    runBine<T, RedOp, ProtoLL>(tid, nthreads, work);
  }
};

template <typename T, typename RedOp>
struct RunWorkColl<ncclFuncBroadcast, T, RedOp, NCCL_ALGO_BINE,
                   NCCL_PROTO_LL128> {
  __device__ __forceinline__ void run(int tid, int nthreads,
                                      struct ncclDevWorkColl *work) {
    runBine<T, RedOp, ProtoLL128>(tid, nthreads, work);
  }
};
