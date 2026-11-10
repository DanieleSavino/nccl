#pragma once

#include "bine_ag.h"   // BineAgGeom, BineAgPre, bineAgPre, ncclPatStep flags (PatUsed, PatSkipped, PatSrcOutput)

// ---- reverse of BineAGAlgorithm ---------------------------------------------
// Index-space butterfly, run backwards. Step s = S-1 .. 0 (span = 2^s blocks):
//   send the half of the current window I do NOT keep  (positions sendBeg+j)
//   recv the half I keep from my dim-s partner and reduce it into my copy (positions keepBeg+j)
// First step (s == S-1): both the sent data and the local operand come straight from sendbuff.
// Later steps: they come from the scratch. Position b lives at scratch[(b & halfMask)*count + col]
// (the kept window is always inside one half of the index space => scratch = nRanks/2 blocks).
// After s == 0 I hold the reduced block of position myIdx, which belongs to rank myIdx:
//   pre.sendPeer < 0 : myIdx == rank, the last reduce writes straight into recvbuff
//   otherwise        : post-step per chunk [send scratch(myIdx) -> pre.recvPeer][recv recvbuff <- pre.sendPeer]
//                      (reverse direction of the allgather pre-step)
// Op order per slot is the same as the allgather: [send P pieces, post][recv P pieces, post].
class BineRSAlgorithm {
  BineAgGeom g; int rank, myIdx; BineAgPre pre;
  int halfMask;
  int stage;                 // 0 butterfly, 1 post-step, 2 done, 3 empty channel, 4 local copy (S == 0)
  int e, half;               // post-step / local-copy cursor, send-recv phase
  int s, gi, k, span, keepBeg, sendBeg, nSlots; size_t total;
 
  __device__ __host__ static size_t bmin(size_t a, size_t b) { return a < b ? a : b; }
 
  __device__ __host__ void enterStep(int step) {
    s = step; span = 1 << s;
    const int blockSize = span << 1;
    const int base = (myIdx / blockSize) * blockSize;
    const bool keepLower = ((myIdx >> s) & 1) == 0;
    keepBeg = keepLower ? base : base + span;        // half containing myIdx: I receive+reduce it
    sendBeg = keepLower ? base + span : base;        // the other half: I give it away
    total  = (size_t)g.nChunks << s;
    nSlots = g.stepSlots(s);
    gi = 0; k = 0; half = 0;
  }
 
public:
  __device__ __host__ BineRSAlgorithm(const BineAgGeom& g_, int rank_, int myIdx_, const BineAgPre& pre_)
      : g(g_), rank(rank_), myIdx(myIdx_), pre(pre_),
        halfMask(g_.S > 0 ? (1 << (g_.S - 1)) - 1 : 0),
        stage(g_.nChunks == 0 ? 3 : (g_.S == 0 ? 4 : 0)), e(0), half(0),
        s(0), gi(0), k(0), span(1), keepBeg(0), sendBeg(0), nSlots(0), total(0) {
    if (stage == 0) enterStep(g.S - 1);
  }
 
  __device__ __host__ int getParallelFactor() const { return 1; }  // single group, like the allgather
 
  __device__ __host__ void getNextOp(struct ncclPatStep* ps) {
    int flags = PatUsed;
    ps->last = 0; ps->nelem = 0; ps->inpIx = ps->outIx = 0;
    ps->sendDim = ps->recvDim = -1; ps->sendOffset = ps->recvOffset = -1;
    ps->stepOffset = 0; ps->postSend = ps->postRecv = 0;
 
    if (stage == 3) {
      flags |= PatSkipped; ps->last = 2;
    } else if (stage == 4) {                         // single rank: sendbuff -> recvbuff
      const size_t eo = g.offset + (size_t)e * g.chunk;
      ps->nelem = (int)bmin(g.chunk, g.end - eo);
      ps->inpIx = eo; ps->outIx = eo; flags |= BineDstFinal;
      if (++e >= g.nChunks) stage = 2;
    } else if (stage == 0) {                         // butterfly, s = S-1 .. 0
      const int len = (int)bmin(g.P, total - (size_t)gi * g.P);
      const size_t q = (size_t)gi * g.P + k;
      const int j = (int)(q % span);
      const size_t eo = g.offset + (q / span) * g.chunk;
      ps->nelem = (int)bmin(g.chunk, g.end - eo);
      const int off = k * g.stride;
      const int post = (k == len - 1) ? 1 : 0;
      const bool first = (s == g.S - 1);
      if (half == 0) {                               // send a piece of the half I give away
        const int blk = sendBeg + j;
        ps->sendDim = s;
        if (first) ps->inpIx = (size_t)blk * g.count + eo;
        else { ps->inpIx = (size_t)(blk & halfMask) * g.count + eo; flags |= PatSrcOutput; }
        ps->sendOffset = off; ps->postSend = post;
      } else {                                       // recv a piece of the half I keep and reduce
        const int blk = keepBeg + j;
        const size_t scIx = (size_t)(blk & halfMask) * g.count + eo;
        ps->recvDim = s; ps->recvOffset = off; ps->postRecv = post;
        if (first) ps->inpIx = (size_t)blk * g.count + eo;          // operand: sendbuff
        else { ps->inpIx = scIx; flags |= PatSrcOutput; }           // operand: scratch (accumulate)
        if (s == 0 && pre.sendPeer < 0) { ps->outIx = eo; flags |= BineDstFinal; }
        else ps->outIx = scIx;
      }
      if (++k == len) {
        k = 0;
        if (half == 0) half = 1;
        else {
          half = 0;
          if (++gi == nSlots) {
            if (s > 0) enterStep(s - 1);
            else if (pre.sendPeer < 0) stage = 2;
            else { stage = 1; e = 0; half = 0; }
          }
        }
      }
    } else {                                         // stage 1: deliver block myIdx to its owner
      const size_t eo = g.offset + (size_t)e * g.chunk;
      ps->nelem = (int)bmin(g.chunk, g.end - eo);
      if (half == 0) {                               // reduced block myIdx -> rank myIdx (= pre.recvPeer)
        ps->sendDim = pre.recvDim; ps->sendOffset = 0; ps->postSend = 1;
        ps->inpIx = (size_t)(myIdx & halfMask) * g.count + eo; flags |= PatSrcOutput;
        half = 1;
      } else {                                       // my reduced block <- pre.sendPeer, straight into recvbuff
        ps->recvDim = pre.sendDim; ps->recvOffset = 0; ps->postRecv = 1;
        ps->outIx = eo; flags |= BineDstFinal | BineRecvCopy;
        half = 0;
        if (++e >= g.nChunks) stage = 2;
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
 
// ---- device entry point -----------------------------------------------------
// scratch: (nRanks/2) * count elements of T per rank, shared by all channels (disjoint columns).
template <typename T, typename RedOp, typename MakeAlgo>
__device__ __forceinline__ void runBineRsPat(int tid, int nthreads, ncclDevWorkColl* work,
                                             T* scratch, MakeAlgo makeAlgo) {
#if __CUDA_ARCH__ >= 600
  using Proto = ProtoSimple<1, 1>;
  ncclBine* bine = &ncclShmem.channel.bine;
  const int S = bine->nDoublingSteps;
  size_t count, channelOffset, channelCount, chunkCount;
  ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T), &count,
                  &channelOffset, &channelCount, &chunkCount);
  const size_t slotElems = ncclShmem.comm.buffSizes[NCCL_PROTO_SIMPLE] / NCCL_STEPS / sizeof(T);
 
  static constexpr int nworkers = NCCL_PAT_NWORKERS;
  struct ncclPatShmem* shmem = (struct ncclPatShmem*)ncclScratchForWarp(0);
  uint64_t pollCount = 0;
  __syncthreads();
  for (int i = tid; i < NCCL_SHMEM_PAT_STEPS; i += nthreads) shmem->patSteps[i].flags = 0;
  if (tid == 0) shmem->localAccSize = 0;
  if (tid == nworkers) shmem->parallelFactor = 0;
  __syncthreads();
 
  if (tid == nworkers) {                       // schedule thread
    BineAgGeom g; g.init(S, count, channelOffset, channelCount, chunkCount, slotElems);
    auto algo = makeAlgo(g);
    shmem->parallelFactor = algo.getParallelFactor();
    int step = 0;
    while (1) {
      struct ncclPatStep* ps = shmem->patSteps + (step % NCCL_SHMEM_PAT_STEPS);
      cuda::atomic_ref<int, cuda::thread_scope_block> poll(ps->flags);
      while (poll.load(cuda::memory_order_acquire) != 0) pollCount++;
      algo.getNextOp(ps);
      int last = ps->last;
      step++;
      if (last == 2) break;
    }
  } else if (tid < nworkers) {                 // workers
    int parallelFactor = 0;
    volatile int* pfPtr = &shmem->parallelFactor;
    while (parallelFactor == 0) parallelFactor = *pfPtr;
    int groupSize = nworkers / (WARP_SIZE * parallelFactor) * WARP_SIZE;
    int group = tid / groupSize, nGroups = nworkers / groupSize, tidInGroup = tid - group * groupSize;
 
    // input = sendbuff, "output" slot of the group = scratch, aux = recvbuff (final result)
    Primitives<T, RedOp, FanSymmetric<1>, 0, Proto, 0> prims(
        tidInGroup, groupSize, (int*)shmem->recvDims, (int*)shmem->sendDims,
        work->sendbuff, scratch, work->redOpArg, group, 0, 0, nullptr, nullptr, 0,
        primsModeBineRs, work->recvbuff);
 
    int step = group;
    while (1) {
      struct ncclPatStep* ps = shmem->patSteps + (step % NCCL_SHMEM_PAT_STEPS);
      cuda::atomic_ref<int, cuda::thread_scope_block> poll(ps->flags);
      while (poll.load(cuda::memory_order_acquire) == 0) pollCount++;
      int last = ps->last;
      prims.bineReduce(ps, shmem);
      if (tidInGroup == 0) poll.store(0, cuda::memory_order_release);
      if (last) break;
      step += nGroups;
    }
  }
#endif
}
 
template <typename T, typename RedOp>
__device__ __forceinline__ void runBineRsSendPat(int tid, int nthreads, ncclDevWorkColl* work) {
  ncclBine* bine = &ncclShmem.channel.bine;
  const int S = bine->nDoublingSteps, rank = ncclShmem.comm.rank;
  // TODO(host): add this field to ncclDevWorkColl and allocate (nRanks/2)*count*sizeof(T) per rank
  T* scratch = (T*)work->bineScratch;
  runBineRsPat<T, RedOp>(tid, nthreads, work, scratch, [&](const BineAgGeom& g) {
    const int myIdx = bineSendIndex(bine->index, rank, S);
    BineAgPre pre = bineAgPre(bine->partners, bine->index, bine->order, rank, S);
    return BineRSAlgorithm(g, rank, myIdx, pre);
  });
}
