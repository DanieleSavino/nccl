#ifndef BINE_HELPER_H_
#define BINE_HELPER_H_

#include "nccl.h"
#include "bine.h"
#include "debug.h"
#include <string.h>
#include <algorithm>

#pragma once
#ifdef __CUDACC__
#define BINE_HD __host__ __device__ __forceinline__
#else
#define BINE_HD inline
#endif

typedef enum {
  BLOCK_BY_BLOCK,
  PERMUTATION,
  DOUBLE_SEND,
  SEND,
  TREE
} ncclBineBufferManagement_t;

static inline const char* ncclBineBufferManagementToString(ncclBineBufferManagement_t val) {
  switch (val) {
    case BLOCK_BY_BLOCK: return "BLOCK_BY_BLOCK";
    case PERMUTATION:    return "PERMUTATION";
    case DOUBLE_SEND:    return "DOUBLE_SEND";
    case SEND:           return "SEND";
    case TREE:           return "TREE";
    default:             return "UNKNOWN";
  }
}

static inline ncclResult_t ncclBineBufferManagementFromString(const char* str, ncclBineBufferManagement_t* val) {
  if (strcmp(str, "BLOCK_BY_BLOCK") == 0) {
    *val = BLOCK_BY_BLOCK;
  } else if (strcmp(str, "PERMUTATION") == 0) {
    *val = PERMUTATION;
  } else if (strcmp(str, "DOUBLE_SEND") == 0) {
    *val = DOUBLE_SEND;
  } else if (strcmp(str, "SEND") == 0) {
    *val = SEND;
  } else if (strcmp(str, "TREE") == 0) {
    *val = TREE;
  } else {
    WARN("Unknown ncclBineBufferManagement_t value: %s", str);
    return ncclInvalidArgument;
  }
  return ncclSuccess;
}

// Returns the number of steps per loop for a given collective.
static int bineNstepsPerLoop(ncclFunc_t coll, int halvingSteps, int doublingSteps) {
  switch (coll) {
  case ncclFuncReduce:
  case ncclFuncBroadcast:
    return halvingSteps;
  case ncclFuncReduceScatter:
  case ncclFuncAllGather:
    return doublingSteps;
  case ncclFuncAllReduce:
    return 2 * doublingSteps;
  default:
    return std::max(halvingSteps, doublingSteps);
  }
}

#ifndef BINE_AR_MAX_FAN
  // Max children per rank = max fan the Primitives templates support.
  // Stock NCCL (device.h) defines NCCL_MAX_DIRECT_ARITY 7; fall back to 7 if
  // the name differs in your tree.
  #if defined(NCCL_MAX_DIRECT_ARITY)
    #define BINE_AR_MAX_FAN NCCL_MAX_DIRECT_ARITY
  #else
    #define BINE_AR_MAX_FAN 7
  #endif
#endif
 
// Root of the tree used by channel `channelId`.
// Spread roots over ranks so that different channels load different links:
// bit-reversal of the channel id (0,16,8,24,... at 32 ranks) keeps the first
// few channels maximally far apart even when nChannels << nRanks.
__host__ __device__ static inline int ncclBineArRoot(int channelId, int nRanks) {
  const int c = channelId % nRanks;
  if (nRanks & (nRanks - 1)) return c;            // non power of two
  int bits = 0;
  while ((1 << bits) < nRanks) ++bits;
  int r = 0;
  for (int i = 0; i < bits; ++i)
    if ((c >> i) & 1) r |= 1 << (bits - 1 - i);
  return r;
}

// Children and parent of `rank` in the broadcast tree rooted at `root`.
// Broadcast convention (same as your broadcast kernel):
//   ncclBineTreeSend(...) >= 0  -> peer is a CHILD at that step
//   ncclBineTreeRecv(...) >= 0  -> peer is the PARENT
// Reduce uses the same tree reversed (recv from children, send to parent).
// children[] is -1 terminated/padded (Primitives counts peers until -1).
template <typename SendTab, typename RecvTab>
__host__ __device__ static inline void
ncclBineArTreeNeighbors(SendTab bineSend, RecvTab bineRecv, int nRanks, int nSteps,
                        int root, int rank, int* children, int* nChildren, int* parent) {
  for (int i = 0; i < BINE_AR_MAX_FAN; ++i) children[i] = -1;
  *nChildren = 0;
  *parent = -1;
  for (int s = 0; s < nSteps; ++s) {
    const int c = ncclBineTreeSend(bineSend, nRanks, nSteps, root, rank, s);
    const int p = ncclBineTreeRecv(bineRecv, nRanks, nSteps, root, rank, s);
    if (c >= 0 && *nChildren < BINE_AR_MAX_FAN) children[(*nChildren)++] = c;
    if (p >= 0) *parent = p;
  }
}

#endif // BINE_HELPER_H_
