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

template <typename T, typename RedOp, typename Proto>
__device__ __forceinline__ void runBineSend(int tid, int nthreads, ncclDevWorkColl *work)
{
  ncclBine *bine = &ncclShmem.channel.bine;
  const int steps = bine->nDoublingSteps;

  if (steps == 0 || !bine->index || !bine->order) {
    runRing<T, RedOp, Proto>(tid, nthreads, work);
    return;
  }

  ssize_t count, gridOffset, channelCount, chunkCount;
  ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T),
                  &count, &gridOffset, &channelCount, &chunkCount);

  if (channelCount == 0) return;

  // ReduceScatter logarithmic reduction happens in-place in the large sendbuff
  T *accumBuf = (T *)work->sendbuff;
  T *outputBuf = (T *)work->recvbuff;
  
  const int rank  = ncclShmem.comm.rank;
  const int myIdx = bine->index[rank];
  
  // POST-STEP mapping: I send the block I hold to the rank that owns it.
  // I receive my block from the rank currently holding it.
  const int redistTo   = myIdx;             // indexRank
  const int redistFrom = bine->order[rank]; // orderRank

  const bool useDirect = (work->direct & (NCCL_P2P_READ | NCCL_P2P_WRITE)) == (NCCL_P2P_READ | NCCL_P2P_WRITE);

  // ---- 1. Logarithmic Halving Phase ----
  // Chunk loop is OUTER to deeply pipeline the network FIFOs
  for (ssize_t elem = 0; elem < channelCount; elem += chunkCount)
  {
    const ssize_t dataOff = gridOffset + elem;
    const int ne = (int)min(chunkCount, channelCount - elem);

    // ReduceScatter halves downwards from steps - 1 to 0
    for (int s = steps - 1; s >= 0; --s)
    {
      const int partner = bine->partners[rank * steps + s];
      if (partner < 0) continue;

      const int span      = 1 << s;
      const int blockSize = span << 1;
      const int base      = (myIdx / blockSize) * blockSize;
      const bool keepLower = ((myIdx >> s) & 1) == 0;

      // In ReduceScatter, if keeping lower, send upper half and recv lower half
      const int sendBeg = keepLower ? base + span : base;
      const int recvBeg = keepLower ? base : base + span;

      int peers[1] = {partner};
      const bool doPost = (s == 0); // Apply final reduction op on the last step

      if (ne == (int)count) {
        // FASTPATH: Un-chunked contiguous transfer of the entire span
        const ssize_t sendOff = (ssize_t)sendBeg * count + dataOff - gridOffset;
        const ssize_t recvOff = (ssize_t)recvBeg * count + dataOff - gridOffset;
        const ssize_t nElemStep = (ssize_t)span * count;

        if (useDirect) {
          Primitives<T, RedOp, FanAsymmetric<1, 1>, 1, Proto, 0> prim(
              tid, nthreads, peers, peers, accumBuf, accumBuf, work->redOpArg);
          prim.directSendFromOutput(sendOff, nElemStep);
          prim.directRecvReduceCopy(recvOff, recvOff, nElemStep, doPost);
        } else {
          Primitives<T, RedOp, FanAsymmetric<1, 1>, 0, Proto, 0> prim(
              tid, nthreads, peers, peers, accumBuf, accumBuf, work->redOpArg);
          prim.sendFromOutput(sendOff, nElemStep);
          prim.recvReduceCopy(recvOff, recvOff, nElemStep, doPost);
        }
      } else {
        // SLOWPATH: Block-by-block pipelined transfer
        for (int j = 0; j < span; ++j) {
          ssize_t sOff = (ssize_t)(sendBeg + j) * count + dataOff;
          ssize_t rOff = (ssize_t)(recvBeg + j) * count + dataOff;

          if (useDirect) {
            Primitives<T, RedOp, FanAsymmetric<1, 1>, 1, Proto, 0> prim(
                tid, nthreads, peers, peers, accumBuf, accumBuf, work->redOpArg);
            prim.directSendFromOutput(sOff, ne);
            prim.directRecvReduceCopy(rOff, rOff, ne, doPost);
          } else {
            Primitives<T, RedOp, FanAsymmetric<1, 1>, 0, Proto, 0> prim(
                tid, nthreads, peers, peers, accumBuf, accumBuf, work->redOpArg);
            prim.sendFromOutput(sOff, ne);
            prim.recvReduceCopy(rOff, rOff, ne, doPost);
          }
        }
      }
    }
  }
  
  // Ensure all threads complete logarithmic reductions before starting the final permutation
  __syncthreads();

  // ---- 2. Post-Step Redistribution Phase ----
  if (redistTo == rank) {
    // Fast path: I already own my final block. Copy from accumBuf to outputBuf.
    T *mySeg = accumBuf + (ssize_t)rank * count;
    if (outputBuf != mySeg) {
      for (ssize_t elem = tid; elem < channelCount; elem += nthreads) {
        outputBuf[gridOffset + elem] = mySeg[gridOffset + elem];
      }
    }
  } else {
    // I must exchange my fully reduced block for my physical block
    int sendPeer[1] = {redistTo};
    int recvPeer[1] = {redistFrom};

    for (ssize_t elem = 0; elem < channelCount; elem += chunkCount) {
      const ssize_t dataOff = gridOffset + elem;
      const int ne = (int)min(chunkCount, channelCount - elem);
      
      const ssize_t sOff = (ssize_t)myIdx * count + dataOff;
      const ssize_t rOff = dataOff; // Fixed: writes cleanly into outputBuf[0 + dataOff]

      if (useDirect) {
        Primitives<T, RedOp, FanAsymmetric<1, 1>, 1, Proto, 0> prim(
            tid, nthreads, recvPeer, sendPeer, accumBuf, outputBuf, work->redOpArg);
        prim.directSend(sOff, rOff, ne);
        prim.directRecv(rOff, ne); 
      } else {
        Primitives<T, RedOp, FanAsymmetric<1, 1>, 0, Proto, 0> prim(
            tid, nthreads, recvPeer, sendPeer, accumBuf, outputBuf, work->redOpArg);
        prim.send(sOff, ne);
        prim.recv(rOff, ne);
      }
    }
  }
}

  template <typename T, typename RedOp, typename Proto>
  __device__ __forceinline__ void runBineBlockByBlock(int tid, int nthreads, ncclDevWorkColl *work)
  {
    ncclBine *bine = &ncclShmem.channel.bine;

    const int steps = bine->nSteps;
    if (steps == 0 || !bine->index || !bine->order)
    {
      runRing<T, RedOp, Proto>(tid, nthreads, work);
      return;
    }

    ssize_t count, gridOffset, channelCount, chunkCount;
    ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T),
                    &count, &gridOffset, &channelCount, &chunkCount);

    if (channelCount == 0)
      return;

    T *accumBuf = (T *)work->sendbuff;
    T *outputBuf = (T *)work->recvbuff;

    const int nRanks = ncclShmem.comm.nRanks;
    const int rank = ncclShmem.comm.rank;
    const int myIndex = bine->index[rank];
    const bool useDirect =
        (work->direct & (NCCL_P2P_READ | NCCL_P2P_WRITE)) == (NCCL_P2P_READ | NCCL_P2P_WRITE);

    int low = 0;
    int high = nRanks;

    for (int step = 0; step < steps; ++step)
    {
      const int mid = (low + high) >> 1;
      const bool keepLower = (myIndex < mid);
      const int span = keepLower ? (high - mid) : (mid - low);

      int partner = -1;
      if (span > 0)
      {
        const int partnerIdx = keepLower ? myIndex + span : myIndex - span;
        const bool partnerInOppositeHalf = keepLower ? (partnerIdx >= mid && partnerIdx < high)
                                                     : (partnerIdx >= low && partnerIdx < mid);
        if (partnerIdx >= 0 && partnerIdx < nRanks && partnerInOppositeHalf)
          partner = bine->order[partnerIdx];
      }

      if (partner >= 0)
      {
        const int recvBegin = keepLower ? low : mid;
        const int sendBegin = keepLower ? mid : low;
        const bool doPost = (step == steps - 1);
        int peers[1] = {partner};

        auto runStepRobust = [&](auto &prims, bool direct)
        {
          for (ssize_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount)
          {
            const ssize_t dataOffset = gridOffset + elemOffset;
            const int nelem = (int)min(chunkCount, channelCount - elemOffset);

            for (int i = 0; i < span; ++i)
            {
              int sIdx = sendBegin + i;
              int rIdx = recvBegin + i;

              int sRank = bine->order[sIdx];
              int rRank = bine->order[rIdx];

              ssize_t sOffset = dataOffset + (ssize_t)sRank * count;
              ssize_t rOffset = dataOffset + (ssize_t)rRank * count;

              if (direct)
                prims.directSendFromOutput(sOffset, nelem);
              else
                prims.sendFromOutput(sOffset, nelem);

              if (direct)
                prims.directRecvReduceCopy(rOffset, rOffset, nelem, doPost);
              else
                prims.recvReduceCopy(rOffset, rOffset, nelem, doPost);
            }
          }
        };

        if (useDirect)
        {
          Primitives<T, RedOp, FanAsymmetric<1, 1>, 1, Proto, 0> prims(
              tid, nthreads, peers, peers, accumBuf, accumBuf, work->redOpArg, 0, 0, 0, work);
          runStepRobust(prims, true);
        }
        else
        {
          Primitives<T, RedOp, FanAsymmetric<1, 1>, 0, Proto, 0> prims(
              tid, nthreads, peers, peers, accumBuf, accumBuf, work->redOpArg, 0, 0, 0, work);
          runStepRobust(prims, false);
        }
      }

      __syncthreads();

      if (keepLower)
        high = mid;
      else
        low = mid;
    }

    const int finalOwnerRank = bine->order[low];
    if (outputBuf != accumBuf)
    {
      const ssize_t chunkBase = (ssize_t)finalOwnerRank * count;
      for (ssize_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount)
      {
        const ssize_t dataOffset = gridOffset + elemOffset;
        const int nelem = (int)min(chunkCount, channelCount - elemOffset);
        const ssize_t dstOffset = dataOffset;
        const ssize_t srcOffset = dataOffset + chunkBase;

        for (ssize_t i = tid; i < (ssize_t)nelem; i += nthreads)
        {
          outputBuf[dstOffset + i] = accumBuf[srcOffset + i];
        }
      }
    }
  }

  template <typename T, typename RedOp, typename Proto>
  __device__ __forceinline__ void runBineDoubleSend(int tid, int nthreads, ncclDevWorkColl *work)
  {
    ncclBine *bine = &ncclShmem.channel.bine;
    const int steps = bine->nDoublingSteps; // Repurposed as total halving steps

    ssize_t count, gridOffset, channelCount, chunkCount;
    ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T),
                    &count, &gridOffset, &channelCount, &chunkCount);

    if (channelCount == 0) return;

    T *accumBuf = (T *)work->sendbuff;
    T *outputBuf = (T *)work->recvbuff;

    const int rank = ncclShmem.comm.rank;
    const int p = ncclShmem.comm.nRanks;
    const bool useDirect = (work->direct & (NCCL_P2P_READ | NCCL_P2P_WRITE)) == (NCCL_P2P_READ | NCCL_P2P_WRITE);

    // ---- 1. Calculate Initial Circular Buffer Start (a_start) ----
    // Matches the inverted distance-halving Bine scatter logic
    int M_down_even = 0, M_up_even = 0;
    for (int g = 0; g < steps; ++g) {
      if (g % 2 == 0) M_down_even += (1 << g);
      else            M_up_even   += (1 << g);
    }
    
    int a_start = (rank % 2 == 0) ? (rank - M_down_even) % p : (rank - M_up_even) % p;
    if (a_start < 0) a_start += p;

    // ---- 2. Logarithmic Distance-Halving Phase ----
    for (ssize_t elem = 0; elem < channelCount; elem += chunkCount)
    {
      const ssize_t dataOff = gridOffset + elem;
      const int ne = (int)min(chunkCount, channelCount - elem);
      int a = a_start;

      for (int s = 0; s < steps; ++s)
      {
        const int partner = bine->dhlvPartners[rank * steps + s];
        if (partner < 0) continue;

        const int g = steps - 1 - s;
        const int span = 1 << g; // Size of the half being exchanged
        const bool send_bottom = ((rank % 2) == (g % 2));

        const int sendStart = (a + (send_bottom ? 0 : span)) % p;
        const int recvStart = (a + (send_bottom ? span : 0)) % p;

        if (send_bottom) {
          a = (a + span) % p;
        }

        int peers[1] = {partner};
        const bool doPost = (s == steps - 1);

        if (ne == (int)count) {
          // FASTPATH: Buffer wraps circularly, max 2 contiguous bulk primitives
          const int s_len1 = min(span, p - sendStart);
          const int s_len2 = span - s_len1;

          const int r_len1 = min(span, p - recvStart);
          const int r_len2 = span - r_len1;

          if (useDirect) {
            Primitives<T, RedOp, FanAsymmetric<1, 1>, 1, Proto, 0> prim(
                tid, nthreads, peers, peers, accumBuf, accumBuf, work->redOpArg);
                
            if (s_len1 > 0) prim.directSendFromOutput((ssize_t)sendStart * count + dataOff, (ssize_t)s_len1 * count);
            if (s_len2 > 0) prim.directSendFromOutput(dataOff, (ssize_t)s_len2 * count);
            
            if (r_len1 > 0) prim.directRecvReduceCopy((ssize_t)recvStart * count + dataOff, (ssize_t)recvStart * count + dataOff, (ssize_t)r_len1 * count, doPost);
            if (r_len2 > 0) prim.directRecvReduceCopy(dataOff, dataOff, (ssize_t)r_len2 * count, doPost);
          } else {
            Primitives<T, RedOp, FanAsymmetric<1, 1>, 0, Proto, 0> prim(
                tid, nthreads, peers, peers, accumBuf, accumBuf, work->redOpArg);
                
            if (s_len1 > 0) prim.sendFromOutput((ssize_t)sendStart * count + dataOff, (ssize_t)s_len1 * count);
            if (s_len2 > 0) prim.sendFromOutput(dataOff, (ssize_t)s_len2 * count);
            
            if (r_len1 > 0) prim.recvReduceCopy((ssize_t)recvStart * count + dataOff, (ssize_t)recvStart * count + dataOff, (ssize_t)r_len1 * count, doPost);
            if (r_len2 > 0) prim.recvReduceCopy(dataOff, dataOff, (ssize_t)r_len2 * count, doPost);
          }
        } else {
          // SLOWPATH: Fallback block-by-block processing if channel slices break contiguity
          if (useDirect) {
            Primitives<T, RedOp, FanAsymmetric<1, 1>, 1, Proto, 0> prim(
                tid, nthreads, peers, peers, accumBuf, accumBuf, work->redOpArg);
            for (int j = 0; j < span; ++j) {
              ssize_t sOff = (ssize_t)((sendStart + j) % p) * count + dataOff;
              ssize_t rOff = (ssize_t)((recvStart + j) % p) * count + dataOff;
              prim.directSendFromOutput(sOff, ne);
              prim.directRecvReduceCopy(rOff, rOff, ne, doPost);
            }
          } else {
            Primitives<T, RedOp, FanAsymmetric<1, 1>, 0, Proto, 0> prim(
                tid, nthreads, peers, peers, accumBuf, accumBuf, work->redOpArg);
            for (int j = 0; j < span; ++j) {
              ssize_t sOff = (ssize_t)((sendStart + j) % p) * count + dataOff;
              ssize_t rOff = (ssize_t)((recvStart + j) % p) * count + dataOff;
              prim.sendFromOutput(sOff, ne);
              prim.recvReduceCopy(rOff, rOff, ne, doPost);
            }
          }
        }
      }
    }

    __syncthreads();

    // ---- 3. Final Copy ----
    // Distance-halving naturally places the final block into the physical rank's position.
    if (outputBuf != accumBuf) {
      const ssize_t srcOff = (ssize_t)rank * count;
      for (ssize_t elem = tid; elem < channelCount; elem += nthreads) {
        outputBuf[gridOffset + elem] = accumBuf[srcOff + gridOffset + elem];
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

    using Proto = ProtoSimple<REDUCESCATTER_CHUNKSTEPS / REDUCESCATTER_SLICESTEPS, REDUCESCATTER_SLICESTEPS>;

    switch (bufferManagement) {
      case BLOCK_BY_BLOCK:
        runBineBlockByBlock<T, RedOp, Proto>(tid, nthreads, work);
        break;
      // case PERMUTATION:
      //   runBinePermutation<T, RedOp, Proto>(tid, nthreads, work);
      //   break;
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
struct RunWorkColl<ncclFuncReduceScatter, T, RedOp, NCCL_ALGO_BINE, NCCL_PROTO_LL>
{
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl *work)
  {
    ncclBine *bine = &ncclShmem.channel.bine;
    const ncclBineBufferManagement_t bufferManagement = bine->bufferManagement;

    switch (bufferManagement) {
      case BLOCK_BY_BLOCK:
        runBineBlockByBlock<T, RedOp, ProtoLL>(tid, nthreads, work);
        break;
      // case PERMUTATION:
      //   runBinePermutation<T, RedOp, ProtoLL>(tid, nthreads, work);
      //   break;
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
struct RunWorkColl<ncclFuncReduceScatter, T, RedOp, NCCL_ALGO_BINE, NCCL_PROTO_LL128>
{
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl *work)
  {
    ncclBine *bine = &ncclShmem.channel.bine;
    const ncclBineBufferManagement_t bufferManagement = bine->bufferManagement;

    switch (bufferManagement) {
      case BLOCK_BY_BLOCK:
        runBineBlockByBlock<T, RedOp, ProtoLL128>(tid, nthreads, work);
        break;
      // case PERMUTATION:
      //   runBinePermutation<T, RedOp, ProtoLL128>(tid, nthreads, work);
      //   break;
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
