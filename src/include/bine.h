#ifndef NCCL_BINE_H_
#define NCCL_BINE_H_

void ncclGetBineTreeDhlv(int nRanks, int steps, int *sendTable, int *recvTable);
void ncclGetBineTreeDdbl(int nRanks, int steps, int *partners, int* index, int* order);

#endif // NCCL_BINE_H_
