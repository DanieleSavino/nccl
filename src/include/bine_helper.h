#ifndef BINE_HELPER_H_
#define BINE_HELPER_H_

#include "nccl.h"
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
  SEND
} ncclBineBufferManagement_t;

static inline const char* ncclBineBufferManagementToString(ncclBineBufferManagement_t val) {
  switch (val) {
    case BLOCK_BY_BLOCK: return "BLOCK_BY_BLOCK";
    case PERMUTATION:    return "PERMUTATION";
    case DOUBLE_SEND:    return "DOUBLE_SEND";
    case SEND:           return "SEND";
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
    return halvingSteps + doublingSteps;
  default:
    return std::max(halvingSteps, doublingSteps);
  }
}


#endif // BINE_HELPER_H_
