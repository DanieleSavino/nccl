#ifndef BINE_UTILS_H
#define BINE_UTILS_H

#define BINE_MAX_STEPS 20

#include <stdlib.h>
#include <limits.h>
#include <stdint.h>
#include <assert.h>
#include <stddef.h>

#if defined(__GNUC__) || defined(__clang__)
#define BINE_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define BINE_UNLIKELY(x) (x)
#endif // defined(__GNUC__) || defined(__clang__)

static int rhos[BINE_MAX_STEPS] = {1, -1, 3, -5, 11, -21, 43, -85, 171, -341,
          683, -1365, 2731, -5461, 10923, -21845, 43691, -87381, 174763, -349525};

static int smallest_negabinary[BINE_MAX_STEPS] = {0, 0, -2, -2, -10, -10, -42, -42,
          -170, -170, -682, -682, -2730, -2730, -10922, -10922, -43690, -43690, -174762, -174762};
static int largest_negabinary[BINE_MAX_STEPS] = {0, 1, 1, 5, 5, 21, 21, 85, 85,
          341, 341, 1365, 1365, 5461, 5461, 21845, 21845, 87381, 87381, 349525};

static inline int pi(int rank, int step, int comm_sz) {
  int dest;

  if((rank & 1) == 0) dest = (rank + rhos[step]) % comm_sz;
  else dest = (rank - rhos[step]) % comm_sz;

  if(dest < 0) dest += comm_sz;

  return dest;
}

static inline void get_indexes_aux(int rank, int step, const int n_steps, const int adj_size, int *bitmap){
  if (step >= n_steps) return;

  int peer;

  for (int s = step; s < n_steps; s++){
    peer = pi(rank, s, adj_size);
    *(bitmap + peer) = 0x1;
    get_indexes_aux(peer, s + 1, n_steps, adj_size, bitmap);
  }
}

static inline void get_indexes(int rank, int step, const int n_steps, const int adj_size, int *bitmap){
  if (step >= n_steps) return;

  int peer = pi(rank, step, adj_size);
  *(bitmap + peer) = 0x1;
  get_indexes_aux(peer, step + 1, n_steps, adj_size, bitmap);
}

static inline int is_power_of_two(int value) {
    return (value & (value - 1)) == 0;
}

static inline int log_2(int value) {
  if(BINE_UNLIKELY(1 > value)) {
    return -1;
  }
  int log = sizeof(int)*8 - 1 - __builtin_clz(value); 
  if (!is_power_of_two(value)) {
    log += 1;
  }
  return log;
}

static inline int next_poweroftwo(int value)
{
  if(BINE_UNLIKELY(0 > value)) {
    return -1;
  }

  if(0 == value) {
    return 1;
  }

  return 1 << (8 * sizeof(int) - __builtin_clz(value));
}

static inline int hibit(int value, int start)
{
  unsigned int mask;

  /* Only look at the part that the caller wanted looking at */
  mask = value & ((1 << start) - 1);

  if(0 == mask) {
    return -1;
  }

  start = (8 * sizeof(int) - 1) - __builtin_clz(mask);

  return start;
}

#endif // BINE_UTILS_H
