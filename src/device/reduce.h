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
    const int nranks = ncclShmem.comm.nRanks;
    const int rank = ncclShmem.comm.rank;
    const int prevRank = ring->userRanks[nranks-1];
    const int root = work->root;
    size_t chunkCount;
    size_t channelCount;
    size_t gridOffset;
    ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T), (size_t*)nullptr, &gridOffset, &channelCount, &chunkCount);
    size_t offset;
    int nelem;

    // Coverity reports that the callee treats &ring->next as an array.  However, due to the use of
    // FanSymmetric<1>, only the first element is ever accessed, so it's fine.
    // coverity[callee_ptr_arith:FALSE]
    Primitives<T, RedOp, FanSymmetric<1>, 0, Proto, 0>
      prims(tid, nthreads, &ring->prev, &ring->next, work->sendbuff, work->recvbuff, work->redOpArg);

    if (prevRank == root) {
      for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
        offset = gridOffset + elemOffset;
        nelem = min(chunkCount, channelCount - elemOffset);
        prims.send(offset, nelem);
      }
    }
    else if (rank == root) {
      for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
        offset = gridOffset + elemOffset;
        nelem = min(chunkCount, channelCount - elemOffset);
        prims.recvReduceCopy(offset, offset, nelem, /*postOp=*/true);
      }
    }
    else {
      for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
        offset = gridOffset + elemOffset;
        nelem = min(chunkCount, channelCount - elemOffset);
        prims.recvReduceSend(offset, nelem);
      }
    }
  }

template <typename T, typename RedOp, typename Proto>
  __device__ __forceinline__ void runBine(int tid, int nthreads, struct ncclDevWorkColl *work)
  {
    ncclBine *bine = &ncclShmem.channel.bine;

    ssize_t gridOffset, channelCount, chunkCount;
    ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T),
                    (ssize_t *)nullptr, &gridOffset, &channelCount, &chunkCount);

    T *inputBuf = (T *)work->sendbuff;
    T *outputBuf = (T *)work->recvbuff;

    const int rank = ncclShmem.comm.rank;
    const int nRanks = ncclShmem.comm.nRanks;
    const int nSteps = bine->nSteps;
    const int root = work->root;
    const bool isRoot = (rank == root);
    const bool useDirect = (work->direct & (NCCL_P2P_READ | NCCL_P2P_WRITE)) != 0;

    bool hasReduced = false;

    __syncthreads();

    for (int step = nSteps - 1; step >= 0; --step)
    {
      // INFO: bine->send/recv hold only the root=0 basis table [nRanks * nSteps];
      // rotate it for the actual root/rank here. Note the roles are swapped
      // vs. the broadcast: this is a reduce walking up the tree, so the
      // "recv" table's peer is who we send to, and vice versa.
      const int sendPeer = ncclBineTreeRecv(bine->recv, nRanks, nSteps, root, rank, step);
      const int recvPeer = ncclBineTreeSend(bine->send, nRanks, nSteps, root, rank, step);

      if (recvPeer >= 0)
      {
        int recvPeers[1] = {recvPeer};

        if (useDirect)
        {
          if (hasReduced)
          {
            Primitives<T, RedOp, FanAsymmetric<1, 0>, /*Direct=*/1, Proto, 0>
                prim(tid, nthreads, recvPeers, /*sendPeers=*/nullptr, outputBuf, outputBuf, work->redOpArg,
                     /*group=*/0, /*connIndexRecv=*/0, /*connIndexSend=*/0, work);
            for (ssize_t e = 0; e < channelCount; e += chunkCount)
            {
              const ssize_t off = gridOffset + e;
              const int ne = (int)min(chunkCount, channelCount - e);

              /**
               * WARN: [HLC]: Had to add this to PrimitivesWithoutDirect struct
               * in primitives.h to make it work on LL and LL128.
               */
              prim.directRecvReduceCopy(off, off, ne, /*postOp=*/isRoot);
            }
          }
          else
          {
            Primitives<T, RedOp, FanAsymmetric<1, 0>, /*Direct=*/1, Proto, 0>
                prim(tid, nthreads, recvPeers, /*sendPeers=*/nullptr, inputBuf, outputBuf, work->redOpArg,
                     /*group=*/0, /*connIndexRecv=*/0, /*connIndexSend=*/0, work);
            for (ssize_t e = 0; e < channelCount; e += chunkCount)
            {
              const ssize_t off = gridOffset + e;
              const int ne = (int)min(chunkCount, channelCount - e);
              prim.directRecvReduceCopy(off, off, ne, /*postOp=*/isRoot);
            }
            hasReduced = true;
          }
        }
        else
        {
          if (hasReduced)
          {
            Primitives<T, RedOp, FanAsymmetric<1, 0>, /*Direct=*/0, Proto, 0>
                prim(tid, nthreads, recvPeers, /*sendPeers=*/nullptr, outputBuf, outputBuf, work->redOpArg,
                     /*group=*/0, /*connIndexRecv=*/0, /*connIndexSend=*/0, work);
            for (ssize_t e = 0; e < channelCount; e += chunkCount)
            {
              const ssize_t off = gridOffset + e;
              const int ne = (int)min(chunkCount, channelCount - e);
              prim.recvReduceCopy(off, off, ne, /*postOp=*/isRoot);
            }
          }
          else
          {
            Primitives<T, RedOp, FanAsymmetric<1, 0>, /*Direct=*/0, Proto, 0>
                prim(tid, nthreads, recvPeers, /*sendPeers=*/nullptr, inputBuf, outputBuf, work->redOpArg,
                     /*group=*/0, /*connIndexRecv=*/0, /*connIndexSend=*/0, work);
            for (ssize_t e = 0; e < channelCount; e += chunkCount)
            {
              const ssize_t off = gridOffset + e;
              const int ne = (int)min(chunkCount, channelCount - e);
              prim.recvReduceCopy(off, off, ne, /*postOp=*/isRoot);
            }
            hasReduced = true;
          }
        }
        continue;
      }

      if (sendPeer >= 0)
      {
        int sendPeers[1] = {sendPeer};

        if (useDirect)
        {
          Primitives<T, RedOp, FanAsymmetric<0, 1>, /*Direct=*/1, Proto, 0>
              prim(tid, nthreads, /*recvPeers=*/nullptr, sendPeers, inputBuf, outputBuf, work->redOpArg,
                   /*group=*/0, /*connIndexRecv=*/0, /*connIndexSend=*/0, work);
          for (ssize_t e = 0; e < channelCount; e += chunkCount)
          {
            const ssize_t off = gridOffset + e;
            const int ne = (int)min(chunkCount, channelCount - e);

            if (hasReduced)
              prim.directSendFromOutput(off, ne);
            else
              prim.directSend(off, off, ne);
          }
        }
        else
        {
          Primitives<T, RedOp, FanAsymmetric<0, 1>, /*Direct=*/0, Proto, 0>
              prim(tid, nthreads, /*recvPeers=*/nullptr, sendPeers, inputBuf, outputBuf, work->redOpArg,
                   /*group=*/0, /*connIndexRecv=*/0, /*connIndexSend=*/0, work);
          for (ssize_t e = 0; e < channelCount; e += chunkCount)
          {
            const ssize_t off = gridOffset + e;
            const int ne = (int)min(chunkCount, channelCount - e);
            if (hasReduced)
              prim.sendFromOutput(off, ne);
            else
              prim.send(off, ne);
          }
        }
        continue;
      }
    }
  }
}

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncReduce, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    using Proto = ProtoSimple<REDUCE_CHUNKSTEPS/REDUCE_SLICESTEPS, REDUCE_SLICESTEPS>;
    runRing<T, RedOp, Proto>(tid, nthreads, work);
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncReduce, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    runRing<T, RedOp, ProtoLL>(tid, nthreads, work);
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncReduce, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL128> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    runRing<T, RedOp, ProtoLL128>(tid, nthreads, work);
  }
};

template <typename T, typename RedOp>
struct RunWorkColl<ncclFuncReduce, T, RedOp, NCCL_ALGO_BINE, NCCL_PROTO_SIMPLE>
{
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl *work)
  {
    using Proto = ProtoSimple<REDUCE_CHUNKSTEPS / REDUCE_SLICESTEPS, REDUCE_SLICESTEPS>;
    runBine<T, RedOp, Proto>(tid, nthreads, work);
  }
};

template <typename T, typename RedOp>
struct RunWorkColl<ncclFuncReduce, T, RedOp, NCCL_ALGO_BINE, NCCL_PROTO_LL>
{
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl *work)
  {
    runBine<T, RedOp, ProtoLL>(tid, nthreads, work);
  }
};

template <typename T, typename RedOp>
struct RunWorkColl<ncclFuncReduce, T, RedOp, NCCL_ALGO_BINE, NCCL_PROTO_LL128>
{
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl *work)
  {
    runBine<T, RedOp, ProtoLL128>(tid, nthreads, work);
  }
};
