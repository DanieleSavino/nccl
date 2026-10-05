#ifndef NCCL_BINE_H_
#define NCCL_BINE_H_

#include "nccl_device/utility.h"
#include "device/bine_utils.h"

#include <stddef.h>

// Root=0 basis: fills sendTable/recvTable, each sized [nRanks * steps].
void ncclGetBineTree(int nRanks, int steps, int *sendTable, int *recvTable);
void ncclGetBineButterflyDdbl(int nRanks, int steps, int *partners, int *index, int *order);
void ncclGetBineButterflyDhlv(int nRanks, int steps, int *partners);

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

NCCL_HOST_DEVICE_INLINE static int bineBitrev(int x, int bits) {
  int y = 0;
  for (int b = 0; b < bits; ++b) if ((x >> b) & 1) y |= 1 << (bits - 1 - b);
  return y;
}
// kernel step s -> generic (paper) step bits-1-s
NCCL_HOST_DEVICE_INLINE static int bineSendPartner(const int* partners, int rank, int s, int bits) {
  return partners[rank * bits + (bits - 1 - s)];
}
// buffer position held by `rank` = reverse(nu(rank))
NCCL_HOST_DEVICE_INLINE static int bineSendIndex(const int* index, int rank, int bits) {
  return bineBitrev(index[rank], bits);
}
// inverse of bineSendIndex: rank that holds position x
NCCL_HOST_DEVICE_INLINE static int bineSendOrder(const int* order, int x, int bits) {
  return order[bineBitrev(x, bits)];
}

#endif // NCCL_BINE_H_
