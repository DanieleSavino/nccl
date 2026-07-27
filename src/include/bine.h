#ifndef NCCL_BINE_H_
#define NCCL_BINE_H_

#include "nccl_device/utility.h"
#include "device/bine_utils.h"

// Root=0 basis: fills sendTable/recvTable, each sized [nRanks * steps].
void ncclGetBineTree(int nRanks, int steps, int *sendTable, int *recvTable);
void ncclGetBineButterflyDdbl(int nRanks, int steps, int *partners, int *index, int *order);

// Rotates a root=0 table to get the peer for arbitrary (root, rank, step).
NCCL_HOST_DEVICE_INLINE int ncclBineTreeLookup(const int *virtualTable, int nRanks, int steps,
                                                   int root, int rank, int step)
{
  const int vrank = pmod(rank - root, nRanks);
  const int vpeer = virtualTable[idx(vrank, step, steps)];
  return (vpeer < 0) ? -1 : pmod(vpeer + root, nRanks);
}

NCCL_HOST_DEVICE_INLINE int ncclBineTreeSend(const int *virtualSend, int nRanks, int steps,
                                                 int root, int rank, int step)
{
  return ncclBineTreeLookup(virtualSend, nRanks, steps, root, rank, step);
}

NCCL_HOST_DEVICE_INLINE int ncclBineTreeRecv(const int *virtualRecv, int nRanks, int steps,
                                                 int root, int rank, int step)
{
  return ncclBineTreeLookup(virtualRecv, nRanks, steps, root, rank, step);
}

#endif // NCCL_BINE_H_
