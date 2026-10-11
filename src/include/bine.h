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

NCCL_HOST_DEVICE_INLINE void ncclBineAgSubtree(int rank, int root, int nRanks, int steps,
                                               int* first, int* count)
{
  const int rel = pmod(rank - root, nRanks);
  if (rel == 0) { *first = root; *count = nRanks; return; }
 
  const uint32_t mask = (steps >= 32) ? 0xFFFFFFFFu : ((1u << steps) - 1u);
  const int m = (int)(0x55555555u & mask);                    // largest positive negabinary value
  const uint32_t nb = (uint32_t)rank2nb_raw(rel <= m ? rel : rel - nRanks) & mask;
 
  const uint32_t lsb = nb & 1u;
  int u = 1;
  while (u < steps && ((nb >> u) & 1u) == lsb) ++u;
  const int L = u - 1;
 
  const uint32_t lowMask = (1u << L) - 1u;
  const int H  = nb2rank_raw((int32_t)(nb & ~lowMask));       // value of the shared high bits
  const int lo = -(int)(0xAAAAAAAAu & lowMask);               // most negative value of L free bits
 
  *first = pmod(root + H + lo, nRanks);
  *count = 1 << L;
}
 
// t-th block (t < nRanks) of a circular range starting at `first`.
NCCL_HOST_DEVICE_INLINE int ncclBineAgBlock(int first, int t, int nRanks)
{
  int b = first + t;
  return (b >= nRanks) ? b - nRanks : b;
}

// Step at which v receives from its parent; -1 for the root.
NCCL_HOST_DEVICE_INLINE int ncclBineTreeStepOf(const int* recvTbl, int nRanks, int steps, int root, int v)
{
  for (int s = 0; s < steps; ++s)
    if (ncclBineTreeRecv(recvTbl, nRanks, steps, root, v, s) >= 0) return s;
  return -1;
}
 
NCCL_HOST_DEVICE_INLINE int ncclBineTreeParentOf(const int* recvTbl, int nRanks, int steps, int root, int v)
{
  const int s = ncclBineTreeStepOf(recvTbl, nRanks, steps, root, v);
  return (s < 0) ? -1 : ncclBineTreeRecv(recvTbl, nRanks, steps, root, v, s);
}
 
// Binomial tree: a node reached at step i owns 2^(s-1-i) ranks (itself included); root owns all.
NCCL_HOST_DEVICE_INLINE int ncclBineTreeSubtreeSize(const int* recvTbl, int nRanks, int steps, int root, int v)
{
  const int s = ncclBineTreeStepOf(recvTbl, nRanks, steps, root, v);
  return (s < 0) ? nRanks : (1 << (steps - 1 - s));
}
 
// Children of v in ascending step order (== descending subtree size). This order is canonical:
// the preorder below, and therefore the on-wire block order, depends on it. Padded with -1 up
// to maxFan so the array can be handed to Primitives.
NCCL_HOST_DEVICE_INLINE void ncclBineTreeChildrenOf(const int* sendTbl, int nRanks, int steps, int root,
                                                    int v, int maxFan, int* children, int* nChildren)
{
  int n = 0;
  for (int s = 0; s < steps && n < maxFan; ++s) {
    const int c = ncclBineTreeSend(sendTbl, nRanks, steps, root, v, s);
    if (c >= 0) children[n++] = c;
  }
  *nChildren = n;
  for (int k = n; k < maxFan; ++k) children[k] = -1;
}
 
// Preorder of the subtree rooted at v (node first, then each child's subtree in ascending step
// order), one block id at a time, no storage. Blocks of any subtree are contiguous in the
// preorder of the whole tree, which is what the tree AllGather / ReduceScatter streams rely on.
struct ncclBinePreorder {
  const int* sendTbl;
  int nRanks, steps, root;
  int node[BINE_MAX_STEPS + 1];
  int nextStep[BINE_MAX_STEPS + 1];
  int top, pending;
 
  NCCL_HOST_DEVICE_INLINE void init(const int* sendTbl_, const int* recvTbl, int nRanks_, int steps_,
                                    int root_, int v) {
    sendTbl = sendTbl_; nRanks = nRanks_; steps = steps_; root = root_;
    pending = v;
    top = 0;
    node[0] = v;
    nextStep[0] = ncclBineTreeStepOf(recvTbl, nRanks, steps, root, v) + 1;
  }
  NCCL_HOST_DEVICE_INLINE bool next(int* b) {
    if (pending >= 0) { *b = pending; pending = -1; return true; }
    while (top >= 0) {
      while (nextStep[top] < steps) {
        const int j = nextStep[top]++;
        const int c = ncclBineTreeSend(sendTbl, nRanks, steps, root, node[top], j);
        if (c < 0) continue;
        ++top;
        node[top] = c;
        nextStep[top] = j + 1;     // c is reached at step j, its children start at j+1
        *b = c;
        return true;
      }
      --top;
    }
    return false;
  }
};

#endif // NCCL_BINE_H_
