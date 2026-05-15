#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <fstream>
#include <sstream>
#include "bine.h"
#include "device/bine_utils.h"

// ------------------------------------------------------------
// HALVING schedule builder
// ------------------------------------------------------------

// Builds the virtual (root=0) halving broadcast schedule.
// For each rank, determines at which step it receives data and to which
// children it forwards in subsequent steps, using negabinary arithmetic.
static void ncclGetVirtualBineTreeDhlv(int nRanks, int steps, int *sendTable, int *recvTable)
{
  if (nRanks <= 0 || !sendTable || !recvTable)
    return;
  assert((1 << steps) == nRanks && "nRanks must be a power of two");

  const size_t elems = (size_t)nRanks * steps;
  std::fill(sendTable, sendTable + elems, -1);
  std::fill(recvTable, recvTable + elems, -1);

  // Count the number of consecutive LSBs that match the least significant bit
  // in the negabinary representation of nb_val. This determines how late in
  // the schedule a rank receives data.
  auto count_trailing_equal_bits = [&](int nb_val) -> int
  {
    if (nb_val == 0)
      return steps;
    const int last_bit = nb_val & 1;
    int count = 1;
    for (int i = 1; i < steps; i++)
    {
      if (((nb_val >> i) & 1) == last_bit)
        count++;
      else
        break;
    }
    return count;
  };

  for (int r = 0; r < nRanks; ++r)
  {
    // Convert rank to its negabinary representation.
    const int r_nb = rank2nb(r, steps);

    // Root (rank 0) sends at every step to a different subtree.
    if (r == 0)
    {
      for (int step = 0; step < steps; step++)
      {
        const int mask     = (1 << (steps - step)) - 1;
        const int child_nb = r_nb ^ mask;
        const int child    = nb2rank(child_nb, steps);
        sendTable[idx(r,     step, steps)] = child;
        recvTable[idx(child, step, steps)] = r;
      }
      continue;
    }

    // For non-root ranks, the receive step is derived from the trailing equal
    // bit count of the negabinary representation.
    const int u         = count_trailing_equal_bits(r_nb);
    const int recv_step = steps - u;

    if (recv_step < 0 || recv_step >= steps)
    {
      fprintf(stderr,
              "Bine: invalid recv_step=%d for rank=%d (u=%d)\n",
              recv_step, r, u);
      continue;
    }

    // Identify the parent by XOR-ing with the step mask.
    const int mask      = (1 << (steps - recv_step)) - 1;
    const int parent_nb = r_nb ^ mask;
    const int parent    = nb2rank(parent_nb, steps);

    recvTable[idx(r,      recv_step, steps)] = parent;
    sendTable[idx(parent, recv_step, steps)] = r;

    // After receiving, forward to children in all subsequent steps.
    for (int send_step = recv_step + 1; send_step < steps; send_step++)
    {
      const int send_mask = (1 << (steps - send_step)) - 1;
      const int child_nb  = r_nb ^ send_mask;
      const int child     = nb2rank(child_nb, steps);
      sendTable[idx(r,     send_step, steps)] = child;
      recvTable[idx(child, send_step, steps)] = r;
    }
  }
}

// Builds the full halving send/recv schedule for all (root, rank) pairs
// by rotating the virtual (root=0) schedule for each possible root.
// Table layout: [root * nRanks + rank][step].
void ncclGetBineTreeDhlv(int nRanks, int steps, int *sendTable, int *recvTable)
{
  if (nRanks <= 0 || !sendTable || !recvTable)
    return;
  assert((1 << steps) == nRanks && "nRanks must be a power of two");

  // Build the virtual schedule first (root=0 basis).
  const size_t vrElems = (size_t)nRanks * steps;
  std::vector<int> virtualSend(vrElems, -1);
  std::vector<int> virtualRecv(vrElems, -1);
  ncclGetVirtualBineTreeDhlv(nRanks, steps, virtualSend.data(), virtualRecv.data());

  // Rotate the virtual schedule for each root by shifting rank indices modulo nRanks.
  const size_t blockStride = (size_t)nRanks * steps;
  const size_t totalElems  = (size_t)nRanks * blockStride;
  std::fill(sendTable, sendTable + totalElems, -1);
  std::fill(recvTable, recvTable + totalElems, -1);

  for (int root = 0; root < nRanks; ++root)
  {
    for (int rank = 0; rank < nRanks; ++rank)
    {
      // Virtual rank is the rank's position relative to the current root.
      const int    vrank   = pmod(rank - root, nRanks);
      const size_t dstBase = ((size_t)root * blockStride) + (size_t)rank * steps;
      const size_t srcBase = (size_t)vrank * steps;

      for (int step = 0; step < steps; ++step)
      {
        const int sendVR = virtualSend[srcBase + step];
        const int recvVR = virtualRecv[srcBase + step];
        // Translate virtual peer ranks back to physical ranks for this root.
        sendTable[dstBase + step] = (sendVR < 0) ? -1 : pmod(sendVR + root, nRanks);
        recvTable[dstBase + step] = (recvVR < 0) ? -1 : pmod(recvVR + root, nRanks);
      }
    }
  }
}

// ------------------------------------------------------------
// DOUBLING schedule builder
// ------------------------------------------------------------

// Builds the partner, index, and order tables for the recursive-doubling phase.
// Each rank is assigned a virtual index via a combined negabinary + Gray-code
// mapping, and partners are matched by flipping one bit of that index per step.
void ncclGetBineButterflyDdbl(int nRanks, int steps, int *partners, int *index, int *order)
{
  if (nRanks <= 0 || !partners)
    return;
  assert((1 << steps) == nRanks && "nRanks must be a power of two");

  const size_t elems = (size_t)nRanks * steps;
  std::fill(partners, partners + elems, -1);

  // Compute the virtual index v(r) for every rank.
  // Even ranks map through the negabinary of (nRanks - r);
  // odd ranks map through the negabinary of r.
  // A Gray-code transform is then applied to produce the final index.
  std::vector<int> v_representation(nRanks);
  for (int r = 0; r < nRanks; ++r)
  {
    int h_val;
    if (r % 2 == 0)
      h_val = rank2nb(nRanks - r, steps); // even rank
    else
      h_val = rank2nb(r, steps);           // odd rank

    v_representation[r] = h_val ^ (h_val >> 1); // Gray-code transform: v(r) = h(r) XOR (h(r) >> 1)
  }

  // Populate index and order lookup tables.
  // index[r]        = virtual index assigned to rank r
  // order[virt_idx] = physical rank that holds virtual index virt_idx
  for (int r = 0; r < nRanks; ++r)
  {
    const int idxVal = v_representation[r]; // in [0, 2^steps)
    index[r] = idxVal;
    order[idxVal] = r;
  }

  // Build the partner table.
  // At each step, rank r is paired with the rank whose virtual index differs
  // from r's by exactly the bit corresponding to that step.
  for (int step = 0; step < steps; ++step)
  {
    const int bit_mask = 1 << step;

    for (int r = 0; r < nRanks; ++r)
    {
      const int target_v = v_representation[r] ^ bit_mask;

      // Find the rank whose virtual index matches the target.
      for (int q = 0; q < nRanks; ++q)
      {
        if (v_representation[q] == target_v)
        {
          partners[idx(r, step, steps)] = q;
          break;
        }
      }
    }
  }
}
