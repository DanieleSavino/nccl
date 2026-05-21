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

// INFO:  [HLC] Added Bine broadcast implementation.
template <typename T, typename RedOp, typename Proto>
__device__ __forceinline__ void runBine(int tid, int nthreads, ncclDevWorkColl *work)
{
  ncclBine *bine = &ncclShmem.channel.bine;

  const int nSteps = bine->nSteps;
  const int rank   = ncclShmem.comm.rank;
  const int nRanks = ncclShmem.comm.nRanks;
  const int root   = work->root;

  // Determine whether zero-copy (direct) paths are available for this operation.
  const bool canDirectRecv = (work->direct & NCCL_P2P_READ)  != 0;
  const bool canDirectSend = (work->direct & NCCL_P2P_WRITE) != 0;

  ssize_t chunkCount;
  ssize_t channelCount;
  ssize_t gridOffset;
  ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T),
                  (ssize_t *)nullptr, &gridOffset, &channelCount, &chunkCount);

  size_t offset;
  int    nelem;

  // The root broadcasts from sendbuff; all other ranks work in-place on recvbuff.
  const T *sendBuff = (rank == root) ? (const T *)work->sendbuff
                                     : (const T *)work->recvbuff;
  T *recvBuff = (T *)work->recvbuff;

  // Base index into the Bine send/recv tables for the (root, rank) pair.
  // Table layout: [root * nRanks + rank][step].
  const size_t rootOffset = ((size_t)root * nRanks + rank) * nSteps;

  for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount)
  {
    offset = gridOffset + elemOffset;
    nelem  = min(chunkCount, channelCount - elemOffset);

    for (int step = 0; step < nSteps; ++step)
    {
      int stepIdx  = rootOffset + step;
      int sendPeer = bine->send[stepIdx];
      int recvPeer = bine->recv[stepIdx];

      // A step with no active peer on either side can be skipped entirely.
      if (sendPeer == -1 && recvPeer == -1)
        continue;

      // Each step is strictly one-directional: exactly one of send/recv is active.
      assert(sendPeer == -1 || recvPeer == -1);

      int recvPeers[1] = {recvPeer};
      int sendPeers[1] = {sendPeer};

      Primitives<T, RedOp, FanAsymmetric<1, 1>, 1, Proto, 0>
          prims(tid, nthreads, recvPeers, sendPeers, sendBuff, recvBuff, work->redOpArg);

      if (recvPeer != -1)
      {
        // Receive step: pull data from the upstream peer.
        if (canDirectRecv)
          prims.directRecv(offset, nelem);
        else
          prims.recv(offset, nelem);
      }
      else if (sendPeer != -1)
      {
        // Send step: push data to the downstream peer.
        if (canDirectSend)
        {
          // If root's sendbuff and recvbuff are aliased, send directly without an
          // intermediate copy; otherwise copy from sendbuff into recvbuff while sending.
          if (rank == root && work->sendbuff == work->recvbuff)
            prims.directSend(offset, offset, nelem);
          else
            prims.directCopySend(offset, offset, nelem);
        }
        else
        {
          // No direct path available — route through the intermediate staging buffer.
          prims.send(offset, nelem);
        }
      }
    }
  }

  // After the Bine steps, the root must copy its own data from sendbuff into recvbuff
  // so that its output is consistent with every other rank's recvbuff.
  if (rank == root && work->sendbuff != work->recvbuff)
  {
    T *src = (T *)work->sendbuff + gridOffset;
    T *dst = (T *)work->recvbuff + gridOffset;
    reduceCopy<COLL_UNROLL, RedOp, T, 0, 1, 1, 0, 1, 1, /*PreOpSrcs=*/0>(
        tid, nthreads, work->redOpArg, false, 1,
        (void **)&src, 1, (void **)&dst, channelCount);
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
