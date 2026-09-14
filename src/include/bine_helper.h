#ifndef BINE_HELPER_H_
#define BINE_HELPER_H_

typedef enum {
  BLOCK_BY_BLOCK,
  PERMUTATION,
  DOUBLE_SEND,
  SEND
} ncclBineBufferManagement_t;

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
