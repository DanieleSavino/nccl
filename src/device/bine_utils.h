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

// TODO: [HLC] Rewrite old CollAlgo comments

/**
 * @brief Negabinary step offsets ρ(k) for BINE butterfly communication.
 *
 * Entry @c k holds the signed offset applied at step @c k of the BINE
 * pattern. Even-ranked processes add this value; odd-ranked processes
 * subtract it. The sequence follows the recurrence ρ(k) = 1 - 2·ρ(k-1).
 */
static int rhos[BINE_MAX_STEPS] = {1, -1, 3, -5, 11, -21, 43, -85, 171, -341,
          683, -1365, 2731, -5461, 10923, -21845, 43691, -87381, 174763, -349525};


/**
 * @brief Smallest integer representable in negabinary with @c k bits.
 *
 * Entry @c k is the minimum value expressible as a @c k-bit negabinary
 * number. Used by @ref in_bine_range for bounds checking before
 * conversion. Odd-indexed entries equal the preceding even-indexed entry
 * because the extra bit in base -2 cannot extend the negative range.
 */
static int smallest_negabinary[BINE_MAX_STEPS] = {0, 0, -2, -2, -10, -10, -42, -42,
          -170, -170, -682, -682, -2730, -2730, -10922, -10922, -43690, -43690, -174762, -174762};

/**
 * @brief Largest integer representable in negabinary with @c k bits.
 *
 * Entry @c k is the maximum value expressible as a @c k-bit negabinary
 * number. Paired with @ref smallest_negabinary for range validation.
 */
static int largest_negabinary[BINE_MAX_STEPS] = {0, 1, 1, 5, 5, 21, 21, 85, 85,
          341, 341, 1365, 1365, 5461, 5461, 21845, 21845, 87381, 87381, 349525};

/**
 * @brief Computes the non-negative remainder of @p a divided by @p b.
 *
 * Unlike the built-in @c % operator, this function always returns a value in
 * the range @c [0, b), even when @p a is negative. This is useful for wrapping
 * grid indices in toroidal (periodic-boundary) cellular automata.
 *
 * @param a Dividend (may be negative).
 * @param b Divisor (must be positive).
 * @return  The mathematical modulus @c a mod @c b, guaranteed non-negative.
 *
 * @note Undefined behaviour if @p b is zero.
 */
static inline int mod(int a, int b) {
    int r = a % b;
    return r < 0 ? r + b : r;
}


/**
 * @brief Tests whether an integer is an exact power of two.
 *
 * Uses the classic bitmask trick: a positive power of two has exactly one bit
 * set, so @c (num & (num-1)) is zero if and only if @p num is a power of two.
 *
 * @param num Value to test.
 * @return    Non-zero (true) if @p num is a power of two; zero (false) otherwise.
 *            Returns zero for non-positive inputs.
 */
static inline int is_pow_2(int num) {
    return num > 0 && (num & (num - 1)) == 0;
}

/**
 * @brief Converts an MPI rank to its full 32-bit negabinary representation.
 *
 * Unmasked variant of @ref rank2nb. Returns the raw signed negabinary
 * integer without truncating to a specific bit width. Useful when the
 * caller needs the full value before applying a size mask.
 *
 * @param rank MPI rank to convert. Must satisfy @c rank <= 0x55555555.
 * @return     Full 32-bit negabinary encoding of @p rank,
 *             or @c -1 if @p rank is out of range.
 */
static inline int rank2nb_raw(int32_t rank) {
    if (rank > 0x55555555)
        return -1;

    const uint32_t mask = 0xAAAAAAAA;
    const int32_t val = (mask + rank) ^ mask;

    return val;
}

/**
 * @brief Converts an MPI rank to its negabinary representation, masked to @p bits.
 *
 * Uses the identity @c nb = ((0xAAAAAAAA + rank) ^ 0xAAAAAAAA) to perform
 * the conversion in O(1) via bitmask arithmetic, then masks the result to
 * the lowest @p bits bits.
 *
 * @param rank MPI rank to convert. Must satisfy @c rank <= 0x55555555.
 * @param bits Number of negabinary bits to retain (determines communicator size).
 * @return     The @p bits-bit negabinary encoding of @p rank,
 *             or @c -1 if @p rank exceeds the representable range.
 */
static inline int rank2nb(int32_t rank, int bits) {
    const int size = (1 << bits);
    return rank2nb_raw(rank) & (size - 1);
}

/**
 * @brief Converts a full 32-bit negabinary value back to an MPI rank.
 *
 * Unmasked inverse of @ref rank2nb_raw. No size masking is applied;
 * the caller is responsible for interpreting the result within the
 * appropriate communicator range.
 *
 * @param nb Full 32-bit negabinary-encoded value.
 * @return   The corresponding signed MPI rank value.
 */
static inline int nb2rank_raw(int32_t nb) {
    const uint32_t mask = 0xAAAAAAAA;
    const int32_t val = (mask ^ nb) - mask;
    return val;
}

/**
 * @brief Converts a negabinary value back to an MPI rank, masked to @p bits.
 *
 * Inverse of @ref rank2nb. Applies @c rank = ((0xAAAAAAAA ^ nb) - 0xAAAAAAAA)
 * and masks to the communicator size.
 *
 * @param nb   Negabinary-encoded value to convert.
 * @param bits Number of bits defining the communicator size (@c 1 << @p bits).
 * @return     The MPI rank corresponding to @p nb within the @p bits-bit range.
 */
static inline int nb2rank(int32_t nb, int bits) {
    const int size = (1 << bits);
    return nb2rank_raw(nb) & (size - 1);
}

/**
 * @brief Computes the BINE butterfly communication partner of @p rank at @p step.
 *
 * At each step of the BINE collective, even-ranked processes communicate
 * with rank @c (rank + ρ(step)) and odd-ranked processes with
 * @c (rank - ρ(step)), both taken modulo @p comm_sz.
 *
 * @param rank    MPI rank of the calling process.
 * @param step    Current butterfly step index (0-based, < @ref NCCL_MAX_BINE_STEPS).
 * @param comm_sz Total number of ranks in the communicator.
 * @return        MPI rank of the communication partner for this step.
 */
static inline int pi(int rank, int step, int comm_sz) {
    int dest;

    if ((rank & 1) == 0)
        dest = (rank + rhos[step]) % comm_sz;
    else
        dest = (rank - rhos[step]) % comm_sz;

    if (dest < 0)
        dest += comm_sz;

    return dest;
}

/**
 * @brief Checks whether an integer falls within the negabinary range for @p nbits bits.
 *
 * Consults the precomputed @ref smallest_negabinary and @ref largest_negabinary
 * tables to determine whether @p x can be represented exactly in negabinary
 * using @p nbits bits.
 *
 * @param x     Integer value to test.
 * @param nbits Number of negabinary bits (index into the range tables).
 * @return      Non-zero (true) if @p x is representable; zero (false) otherwise.
 */
static inline int in_bine_range(int x, uint32_t nbits) {
    return x >= smallest_negabinary[nbits] && x <= largest_negabinary[nbits];
}

/**
 * @brief Reverses all 32 bits of @p x.
 *
 * Performs bit-reversal using five rounds of paired swap, each handling
 * a progressively coarser granularity (1-bit, 2-bit, 4-bit, 8-bit, 16-bit).
 * Used by @ref nb_to_nu and @ref remap_rank to reorder bit patterns
 * after Gray-code conversion.
 *
 * @param x 32-bit value to reverse.
 * @return  @p x with its bits in reversed order.
 */
static inline uint32_t reverse32(uint32_t x) {
    x = ((x >> 1) & 0x55555555u) | ((x & 0x55555555u) << 1);
    x = ((x >> 2) & 0x33333333u) | ((x & 0x33333333u) << 2);
    x = ((x >> 4) & 0x0f0f0f0fu) | ((x & 0x0f0f0f0fu) << 4);
    x = ((x >> 8) & 0x00ff00ffu) | ((x & 0x00ff00ffu) << 8);
    x = ((x >> 16) & 0xffffu)    | ((x & 0xffffu) << 16);
    return x;
}

/**
 * @brief Computes the ceiling log₂ of @p value.
 *
 * Uses @c __builtin_clz to locate the most-significant set bit, yielding
 * @c floor(log2(value)). If @p value is not an exact power of two the
 * result is incremented by one, giving the ceiling.
 *
 * @param value Positive integer whose ceiling log₂ is required.
 * @return      ⌈log₂(@p value)⌉.
 *
 * @note Behaviour is undefined if @p value is zero (due to @c __builtin_clz).
 */
static inline int ceil_log_2(int value) {
    int log = sizeof(int) * 8 - 1 - __builtin_clz(value);
    if (!is_pow_2(value)) {
        log += 1;
    }
    return log;
}

/**
 * @brief Converts a negabinary value to its natural-unit (nu) index within @p size.
 *
 * Applies a Gray-code decode followed by a @ref reverse32 and a right-shift
 * to map a negabinary bit pattern onto a contiguous natural-unit index in
 * @c [0, size).
 *
 * @param nb   Negabinary-encoded value to convert.
 * @param size Communicator size; must be a power of two.
 * @return     Natural-unit index of @p nb within the communicator.
 */
static inline uint32_t nb_to_nu(uint32_t nb, uint32_t size) {
    return reverse32(nb ^ (nb >> 1)) >> (32 - ceil_log_2(size));
}

/**
 * @brief Computes the natural-unit (nu) index of @p rank in a communicator of @p size.
 *
 * A rank may map to one or two candidate negabinary representations when the
 * communicator size is not a power of two. This function evaluates both
 * candidates and selects the one with the smaller natural-unit index,
 * ensuring a consistent total ordering across all ranks.
 *
 * @param rank MPI rank whose natural-unit index is required.
 * @param size Total number of ranks in the communicator.
 * @return     Natural-unit index of @p rank.
 *
 * @note Asserts that at least one valid negabinary candidate exists.
 */
static inline uint32_t nu(uint32_t rank, uint32_t size) {
    uint32_t nba = UINT32_MAX, nbb = UINT32_MAX;
    size_t num_bits = ceil_log_2(size);
    if (rank % 2) {
        if (in_bine_range(rank, num_bits))
            nba = rank2nb_raw(rank);
        if (in_bine_range(rank - size, num_bits))
            nbb = rank2nb_raw(rank - size);
    } else {
        if (in_bine_range(-rank, num_bits))
            nba = rank2nb_raw(-rank);
        if (in_bine_range(-rank + size, num_bits))
            nbb = rank2nb_raw(-rank + size);
    }
    assert(nba != UINT32_MAX || nbb != UINT32_MAX);

    if (nba == UINT32_MAX)        return nb_to_nu(nbb, size);
    if (nbb == UINT32_MAX)        return nb_to_nu(nba, size);

    /* Both candidates valid — pick the one with the smaller nu index. */
    int nu_a = nb_to_nu(nba, size);
    int nu_b = nb_to_nu(nbb, size);
    return nu_a < nu_b ? nu_a : nu_b;
}

/**
 * @brief Resolves the canonical negabinary representation of @p rank with wraparound.
 *
 * When a rank has two valid negabinary representations (possible when the
 * communicator size is not a power of two), this function selects the
 * canonical one by inspecting the most-significant bit of each candidate.
 * The candidate whose MSB (within the @c num_bits window) is set is preferred.
 *
 * @param num_ranks Total number of ranks in the communicator.
 * @param rank      MPI rank to resolve.
 * @return          Canonical negabinary value for @p rank.
 *
 * @note Asserts that at least one valid negabinary candidate exists.
 */
static inline uint32_t nb2rank_wrap(uint32_t num_ranks, uint32_t rank) {
    nb2rank_raw(rank);
    uint32_t nba = UINT32_MAX, nbb = UINT32_MAX;
    size_t num_bits = ceil_log_2(num_ranks);
    if (rank % 2) {
        if (in_bine_range(rank, num_bits))
            nba = rank2nb_raw(rank);
        if (in_bine_range(rank - num_ranks, num_bits))
            nbb = rank2nb_raw(rank - num_ranks);
    } else {
        if (in_bine_range(-rank, num_bits))
            nba = rank2nb_raw(-rank);
        if (in_bine_range(-rank + num_ranks, num_bits))
            nbb = rank2nb_raw(-rank + num_ranks);
    }
    assert(nba != UINT32_MAX || nbb != UINT32_MAX);

    if (nba == UINT32_MAX) return nbb;
    if (nbb == UINT32_MAX) return nba;

    /* Both valid — prefer the candidate whose MSB within num_bits is set. */
    if (nba & (0x80000000 >> (32 - num_bits)))
        return nba;
    else
        return nbb;
}

/**
 * @brief Remaps an MPI rank to its BINE reordered index within a communicator.
 *
 * Full remapping pipeline:
 *   1. Resolve the canonical negabinary value via @ref nb2rank_wrap.
 *   2. Apply a Gray-code decode (@c nb ^ (nb >> 1)).
 *   3. Bit-reverse the result with @ref reverse32.
 *   4. Shift down to retain only the @c num_bits significant bits.
 *
 * The resulting index places ranks in the order expected by the BINE
 * butterfly schedule.
 *
 * @param num_ranks Total number of ranks in the communicator.
 * @param rank      MPI rank to remap.
 * @return          BINE-reordered index of @p rank in @c [0, num_ranks).
 */
static inline uint32_t remap_rank(uint32_t num_ranks, uint32_t rank) {
    uint32_t remap_rank = nb2rank_wrap(num_ranks, rank);
    remap_rank = remap_rank ^ (remap_rank >> 1);
    size_t num_bits = ceil_log_2(num_ranks);
    remap_rank = reverse32(remap_rank) >> (32 - num_bits);
    return remap_rank;
}

/**
 * @brief Computes the @c n-bit Mersenne number @c 2^(n+1) - 1.
 *
 * Returns a bitmask with the lowest @c n+1 bits set. Used by
 * @ref remap_ddbl during the double-binary remapping to progressively
 * mask out processed bit positions.
 *
 * @param n Bit index (0-based).
 * @return  @c (1 << (n+1)) - 1.
 */
static inline uint32_t mersenne(int n) {
    return (1UL << (n + 1)) - 1;
}

/**
 * @brief Remaps a value using the double-binary (Mersenne XOR) scheme.
 *
 * Iteratively extracts the highest set bit of @p num, toggles the
 * corresponding bit in the result, then XORs @p num with the Mersenne
 * number for that bit position to remove the contribution of all lower
 * bits. Continues until @p num is zero.
 *
 * @param num Non-negative integer to remap.
 * @return    Double-binary remapped value of @p num.
 */
static inline int remap_ddbl(uint32_t num) {
    int remapped = 0;
    while (num > 0) {
        int k = 31 - __builtin_clz(num);
        remapped ^= (0x1 << k);
        num ^= mersenne(k);
    }
    return remapped;
}

static inline int pmod(int x, int m)
{
  int r = x % m;
  return r < 0 ? r + m : r;
}

static inline size_t idx(int vr, int step, int steps)
{
  return (size_t)vr * steps + step;
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
