/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "comm.h"
#include "transport.h"
#include "bootstrap.h"
#include "bine.h"
#include <nccl.h>
#include <sstream>
#include "bine_helper.h"

NCCL_PARAM(MultiSegmentRegister, "MULTI_SEGMENT_REGISTER", 1);

ncclResult_t ncclTransportRingConnect(struct ncclComm* comm) {
  struct ringConnInfo {
    bool useNetPXN;
    bool useGdr;
  };
  struct ringConnInfo* ringInfo = NULL;
  ncclResult_t ret = ncclSuccess;
  if (comm && comm->nRanks > 1) {
    comm->useGdr = true;
    comm->useNetPXN = false;
    for (int c = 0; c < comm->nChannels; c++) {
      struct ncclChannel* channel = comm->channels + c;
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &channel->ring.prev, 1, &channel->ring.next, 0), ret, fail);
    }
    NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_RING], 0), ret, fail);
    if (ncclParamLocalRegister() || ncclParamGraphRegister()) {
      NCCLCHECK(ncclCalloc(&ringInfo, comm->nRanks));
      ringInfo[comm->rank].useGdr = comm->useGdr;
      ringInfo[comm->rank].useNetPXN = comm->useNetPXN;
      NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, ringInfo, sizeof(struct ringConnInfo)), ret, fail);
      for (int i = 0; i < comm->nRanks; ++i) {
        if (!ringInfo[i].useGdr) comm->useGdr = false;
        if (ringInfo[i].useNetPXN) comm->useNetPXN = true;
        if (comm->useGdr == false && comm->useNetPXN == true) break;
      }
    }
    INFO(NCCL_INIT, "Connected all rings, use ring PXN %d GDR %d", comm->useNetPXN, comm->useGdr);
  }
exit:
  free(ringInfo);
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclTransportTreeConnect(struct ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;
  if (comm && comm->nRanks > 1) {
    // Connect Trees
    for (int c = 0; c < comm->nChannels; c++) {
      struct ncclChannel* channel = comm->channels + c;
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, NCCL_MAX_TREE_ARITY, channel->tree.down, 1, &channel->tree.up, 0), ret, fail);
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &channel->tree.up, NCCL_MAX_TREE_ARITY, channel->tree.down, 0), ret, fail);
    }
    NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_TREE], 0), ret, fail);
    INFO(NCCL_INIT, "Connected all trees");
  }
exit:
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclTransportPatConnect(struct ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;
  if (comm && comm->nRanks > 1) {
    for (int mask=1; mask<comm->nRanks; mask<<=1) {
      int prevPeer = (comm->rank + mask) % comm->nRanks;
      int nextPeer = (comm->rank + comm->nRanks - mask) % comm->nRanks;
      for (int c = 0; c < comm->nChannels; c++) {
        NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &prevPeer, 1, &nextPeer, 0), ret, fail); // ReduceScatter
      }
      NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_TREE], 0), ret, fail);
      for (int c = 0; c < comm->nChannels; c++) {
        NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &nextPeer, 1, &prevPeer, 0), ret, fail); // AllGather
      }
      NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_TREE], 0), ret, fail);
    }
    INFO(NCCL_INIT, "Connected binomial trees");
  }
exit:
  return ret;
fail:
  goto exit;
}

// INFO: [HLC] Added bine connect.
// FIXME: [HLC] Connect actual pairs.
ncclResult_t ncclTransportBineConnect(struct ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;
  if (comm == nullptr || comm->nRanks <= 1) return ret;
 
  const int nRanks = comm->nRanks;
  bool anySchedules = false;
  bool connectedAnyPeers = false;
  int connectedChannels = 0;
  struct ncclTopoGraph* setupGraph = nullptr;
 
  INFO(NCCL_INIT, "BINE connect rank %d cudaDev %d nRanks %d nChannels %d",
       comm->rank, comm->cudaDev, comm->nRanks, comm->nChannels);
 
  auto isBineEnabledFor = [&](ncclFunc_t func) {
    for (int proto = 0; proto < NCCL_NUM_PROTOCOLS; ++proto) {
      if (comm->bandwidths[func][NCCL_ALGO_BINE][proto] > 0.0f) return true;
    }
    return false;
  };
 
  const bool bcastEnabled  = isBineEnabledFor(ncclFuncBroadcast);
  const bool reduceEnabled = isBineEnabledFor(ncclFuncReduce);
  const bool rsEnabled     = isBineEnabledFor(ncclFuncReduceScatter);
  const bool agEnabled     = isBineEnabledFor(ncclFuncAllGather);
  const bool arEnabled     = isBineEnabledFor(ncclFuncAllReduce);
 
  for (int c = 0; c < comm->nChannels; ++c) {
    struct ncclChannel* channel = comm->channels + c;
    const int steps = channel->bine.nSteps;
    const int doublingSteps = channel->bine.nDoublingSteps;
 
    INFO(NCCL_INIT,
         "BINE channel %d steps %d doublingSteps %d bufferManagement %s partner=%p dhlvPartner=%p index=%p order=%p",
         c, steps, doublingSteps,
         ncclBineBufferManagementToString(channel->bine.bufferManagement),
         channel->binePartner, channel->dhlvBinePartner, channel->bineIndex, channel->bineOrder);
 
    if (steps == 0 && doublingSteps == 0) {
      INFO(NCCL_INIT, "BINE channel %d has zero steps and will be skipped", c);
      continue;
    }
 
    const bool hasStepTables = steps > 0 && channel->bineSend && channel->bineRecv;
    const bool hasDoublingTables =
        doublingSteps > 0 && (channel->binePartner != nullptr || channel->dhlvBinePartner != nullptr);
 
    if (steps > 0 && !hasStepTables) {
      WARN("BINE send/recv tables are missing for channel %d", c);
      ret = ncclInternalError; goto fail;
    }
    if (doublingSteps > 0 && !hasDoublingTables) {
      WARN("BINE partner tables are missing for channel %d", c);
      ret = ncclInternalError; goto fail;
    }
    if (doublingSteps > 0 && (1 << doublingSteps) != nRanks) {
      WARN("BINE doubling schedule requires nRanks == 2^doublingSteps (nRanks %d, doublingSteps %d, channel %d)",
           nRanks, doublingSteps, c);
      ret = ncclInternalError; goto fail;
    }
 
    const ncclBineBufferManagement_t buffMan = channel->bine.bufferManagement;
 
    if (buffMan == SEND && doublingSteps > 0 &&
        (channel->bineIndex == nullptr || channel->bineOrder == nullptr)) {
      WARN("BINE index/order tables are missing for SEND on channel %d", c);
      ret = ncclInternalError; goto fail;
    }
 
    // ---- AllReduce variants ----
    // RSAG: RS halving + AG doubling over binePartner[] only (SEND buffers).
    const bool arRsagOk =
        arEnabled && buffMan == SEND && doublingSteps > 0 && channel->binePartner != nullptr;
    // TREE: pipelined reduce+bcast on one fixed-root tree per channel.
    const bool arTreeOk = arEnabled && hasStepTables && steps <= BINE_AR_MAX_FAN;
    const bool agTreeOk = agEnabled && buffMan == TREE && hasStepTables &&
                           steps <= BINE_AR_MAX_FAN && (1 << steps) == nRanks;

    const bool rsTreeOk = rsEnabled && buffMan == TREE && hasStepTables &&
                           steps <= BINE_AR_MAX_FAN && (1 << steps) == nRanks;
    // Legacy all-roots tree for AR, only if neither new variant covers it.
    const bool arNeedsTree = arEnabled && !arRsagOk && !arTreeOk;
 
    const bool enableBroadcastPhase = hasStepTables && (bcastEnabled || arNeedsTree);
    const bool enableReducePhase    = hasStepTables && (reduceEnabled || rsEnabled || arNeedsTree);
    const bool enableDoublingPhase  = hasDoublingTables && (rsEnabled || agEnabled || arEnabled);
    const bool enableRedistribution = enableDoublingPhase && buffMan == SEND && (agEnabled || rsEnabled);
 
    if (!(enableBroadcastPhase || enableReducePhase || enableDoublingPhase || arTreeOk || agTreeOk || rsTreeOk)) {
      INFO(NCCL_INIT, "BINE channel %d: all phases disabled", c);
      continue;
    }
    anySchedules = true;
 
    std::vector<int> sendPeers, recvPeers;
    auto addPeer = [&](int peer, bool sendToPeer, bool recvFromPeer) {
      if (peer < 0 || peer == comm->rank) return;
      if (sendToPeer) sendPeers.push_back(peer);
      if (recvFromPeer) recvPeers.push_back(peer);
    };
 
    // ---- Tree (halving) phase for bcast / reduce / legacy AR: all roots ----
    if (enableBroadcastPhase || enableReducePhase) {
      const int rank = comm->rank;
      for (int root = 0; root < nRanks; ++root) {
        for (int step = 0; step < steps; ++step) {
          int sendPeer = ncclBineTreeSend(channel->bineSend, nRanks, steps, root, rank, step);
          int recvPeer = ncclBineTreeRecv(channel->bineRecv, nRanks, steps, root, rank, step);
          if (enableBroadcastPhase) {
            addPeer(sendPeer, true, false);
            addPeer(recvPeer, false, true);
          }
          if (enableReducePhase) {
            addPeer(sendPeer, false, true);
            addPeer(recvPeer, true, false);
          }
        }
      }
    }
 
    // ---- AR TREE variant: only this channel's root. Bidirectional on every
    //      neighbour (reduce uses child->parent, bcast uses parent->child). ----
    if (arTreeOk || agTreeOk || rsTreeOk) {
      const int root = ncclBineArRoot(c, nRanks);
      int children[BINE_AR_MAX_FAN];
      int nChildren, parent;
      ncclBineArTreeNeighbors(channel->bineSend, channel->bineRecv, nRanks, steps, root,
                              comm->rank, children, &nChildren, &parent);
      for (int i = 0; i < nChildren; ++i) addPeer(children[i], true, true);
      addPeer(parent, true, true);
      INFO(NCCL_INIT, "BINE AR tree channel %d root %d rank %d children %d parent %d",
           c, root, comm->rank, nChildren, parent);
    }
 
    // ---- Doubling / butterfly phase ----
    if (enableDoublingPhase) {
      if (enableRedistribution) {
        const int redistTo   = bineSendOrder(channel->bineOrder, comm->rank, doublingSteps);
        const int redistFrom = bineSendIndex(channel->bineIndex, comm->rank, doublingSteps);
        if (redistTo != comm->rank) {
          addPeer(redistTo,   true, true);
          addPeer(redistFrom, true, true);
        }
      }
      if (channel->binePartner != nullptr) {
        for (int s = 0; s < doublingSteps; ++s)
          addPeer(bineSendPartner(channel->binePartner, comm->rank, s, doublingSteps), true, true);
      }
      if (channel->dhlvBinePartner != nullptr && buffMan == DOUBLE_SEND) {
        for (int step = 0; step < doublingSteps; ++step)
          addPeer(channel->dhlvBinePartner[comm->rank * doublingSteps + step], true, true);
      }
    }
 
    auto dedup = [](std::vector<int>& peers) {
      std::sort(peers.begin(), peers.end());
      peers.erase(std::unique(peers.begin(), peers.end()), peers.end());
    };
    dedup(sendPeers);
    dedup(recvPeers);
 
    if (!sendPeers.empty() || !recvPeers.empty()) {
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c,
          (int)recvPeers.size(), recvPeers.empty() ? nullptr : recvPeers.data(),
          (int)sendPeers.size(), sendPeers.empty() ? nullptr : sendPeers.data(),
          0), ret, fail);
      connectedAnyPeers = true;
      connectedChannels += 1;
    } else {
      INFO(NCCL_INIT, "BINE channel %d produced no distinct peers after dedup", c);
    }
  }
 
  if (!anySchedules) goto exit;
  if (!connectedAnyPeers) {
    WARN("BINE schedule tables produced no peer connections");
    ret = ncclInternalError; goto fail;
  }
 
  setupGraph = &comm->graphs[NCCL_ALGO_TREE];
  if (setupGraph->nChannels <= 0) {
    WARN("BINE connect rank %d: topo graph for P2P setup has no channels", comm->rank);
    ret = ncclInternalError; goto fail;
  }
  NCCLCHECKGOTO(ncclTransportP2pSetup(comm, setupGraph, 0), ret, fail);
  INFO(NCCL_INIT, "Connected BINE peers on %d channels", connectedChannels);
 
exit:
  if (!anySchedules)
    INFO(NCCL_INIT, "BINE connect rank %d disabled: all phases removed", comm->rank);
  return ret;
fail:
  goto exit;
}
