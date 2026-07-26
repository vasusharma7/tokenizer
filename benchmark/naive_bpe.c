/**
 * naive_bpe.c
 * Deliberately naive Byte-Pair Encoding (BPE) implementation in C.
 *
 * Demonstrates the three bottlenecks described in the canonical BPE explanation:
 *   Bottleneck 1 — Full O(N) sequence scan per merge step → O(N²) total
 *   Bottleneck 2 — Array shifting on every merge (memmove of all subsequent elements)
 *   Bottleneck 3 — Lots of heap churn from strdup / mallocs
 *
 * Usage:
 *   cc -O0 -g -o naive_bpe naive_bpe.c
 *   ./naive_bpe <input.txt> [num_merges] [repeat]
 */

#define _GNU_SOURCE
#include <ctype.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

/* ==================================================================
 *  Merge table: pair (left, right) → new token ID + merge rank
 *
 *  Stored as a flat 2D array indexed by [left * vocab_cap + right].
 *  O(1) lookup — one multiplication, one memory access.
 *  Still naive: allocates a full N×N matrix when most pairs never occur.
 * ================================================================== */

static int *merge_rank   = NULL;   /* -1 = not mergeable; 0..M-1 = priority */
static int *merge_new_id = NULL;   /* replacement token ID */
static int  vocab_capacity = 0;    /* dimension of the square matrix */
static int  next_new_id = 256;     /* IDs 0–255 reserved for raw bytes */
static int  num_merges_total = 0;

/* Initialize (or re-init) the table for up to `max_id` distinct tokens. */
static void merge_table_init(int max_id) {
    size_t n = (size_t)max_id * max_id;
    merge_rank   = malloc(n * sizeof(int));
    merge_new_id = malloc(n * sizeof(int));
    if (!merge_rank || !merge_new_id) { perror("malloc"); exit(1); }
    for (size_t i = 0; i < n; i++) merge_rank[i] = -1;
    vocab_capacity = max_id;
    next_new_id = 256;
    num_merges_total = 0;
}

/* Lookup rank for pair (left, right). Returns -1 if not mergeable. */
static int lookup_rank(int left, int right) {
    if (left >= vocab_capacity || right >= vocab_capacity) return -1;
    return merge_rank[left * vocab_capacity + right];
}

/* Lookup new_id for pair (left, right). Returns -1 if not mergeable. */
static int lookup_new_id(int left, int right) {
    if (left >= vocab_capacity || right >= vocab_capacity) return -1;
    return merge_new_id[left * vocab_capacity + right];
}

/* Add a new merge rule. */
static void add_merge(int left, int right, int rank) {
    int idx = left * vocab_capacity + right;
    merge_rank[idx]   = rank;
    merge_new_id[idx] = next_new_id++;
    num_merges_total++;
}

/* Free merge table */
static void free_merge_table(void) {
    free(merge_rank);
    free(merge_new_id);
    merge_rank = NULL;
    merge_new_id = NULL;
    vocab_capacity = 0;
    next_new_id = 256;
    num_merges_total = 0;
}

/* ==================================================================
 *  Timing
 * ================================================================== */

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ==================================================================
 *  Naive BPE — TRAINING
 *
 *  Given raw text bytes, learn M merge rules.
 *
 *  Algorithm per iteration:
 *    1. Scan the entire sequence to count all adjacent pairs
 *    2. Find the most frequent pair
 *    3. Add it as a new merge rule
 *    4. Merge ALL occurrences of that pair in the sequence
 *       (shift the array left after each merge — O(N) per merge!)
 *
 *  This is SLOW because:
 *    - Step 1 is O(N) per iteration → O(N·M) total just for scanning
 *    - Step 4 does memmove for each merge → O(N²) in the worst case
 *    - We re-allocate the sequence array via realloc repeatedly
 * ================================================================== */

typedef struct {
    int *tokens;      /* dynamic array of token IDs */
    int   length;     /* current number of tokens */
    int   capacity;   /* allocated capacity */
} seq_t;

static void seq_init(seq_t *s, int cap_hint) {
    s->length   = 0;
    s->capacity = cap_hint > 0 ? cap_hint : 1024;
    s->tokens   = malloc(s->capacity * sizeof(int));
    if (!s->tokens) { perror("malloc"); exit(1); }
}

static void seq_append(seq_t *s, int id) {
    if (s->length >= s->capacity) {
        s->capacity *= 2;
        s->tokens = realloc(s->tokens, s->capacity * sizeof(int));
        if (!s->tokens) { perror("realloc"); exit(1); }
    }
    s->tokens[s->length++] = id;
}

/* Remove element at index `pos` by shifting everything left.
 * This is Bottleneck #2: O(N) memmove per call! */
static void seq_remove_at(seq_t *s, int pos) {
    if (pos < 0 || pos >= s->length) return;
    /* Shift all elements after pos one position left */
    memmove(&s->tokens[pos], &s->tokens[pos + 1],
            (s->length - pos - 1) * sizeof(int));
    s->length--;
}

/* Replace two adjacent elements (at pos, pos+1) with a single new_id.
 * This is the core merge operation — does a remove + in-place set.
 * Shift cost: O(N) due to seq_remove_at. */
static void seq_merge_at(seq_t *s, int pos, int new_id) {
    s->tokens[pos] = new_id;
    seq_remove_at(s, pos + 1);
}

/* Count all adjacent pairs in the sequence using a flat 2D count array.
 * This is O(N) — one pass, no inner loop. Still naive: we allocate a full
 * vocab_capacity × vocab_capacity matrix even though most entries stay 0. */
static int *pair_counts = NULL;   /* reusable flat 2D array for pair counting */
static int  pair_counts_cap = 0;

static void pair_counts_init(int max_id) {
    size_t n = (size_t)max_id * max_id;
    pair_counts = malloc(n * sizeof(int));
    if (!pair_counts) { perror("malloc"); exit(1); }
    pair_counts_cap = max_id;
}

static void pair_counts_reset(void) {
    size_t n = (size_t)pair_counts_cap * pair_counts_cap;
    memset(pair_counts, 0, n * sizeof(int));
}

static void pair_counts_free(void) {
    free(pair_counts);
    pair_counts = NULL;
    pair_counts_cap = 0;
}

/* Find the most frequent adjacent pair in the sequence.
 * Uses the flat count array — O(N) to count, O(vocab²) to find max.
 * Returns the position of the first occurrence, sets out_left/out_right. */
static int find_most_frequent_pair(const seq_t *s, int *out_left, int *out_right) {
    pair_counts_reset();
    int cap = pair_counts_cap;

    /* Count every adjacent pair — O(N) */
    for (int i = 0; i < s->length - 1; i++) {
        int l = s->tokens[i];
        int r = s->tokens[i + 1];
        if (l < cap && r < cap)
            pair_counts[l * cap + r]++;
    }

    /* Find the pair with the highest count — O(vocab²) */
    int best_l = -1, best_r = -1, best_count = 0;
    for (int l = 0; l < cap; l++) {
        for (int r = 0; r < cap; r++) {
            int c = pair_counts[l * cap + r];
            if (c > best_count) {
                best_count = c;
                best_l = l;
                best_r = r;
            }
        }
    }

    if (best_l < 0) return -1;  /* no pairs found */

    *out_left  = best_l;
    *out_right = best_r;

    /* Find FIRST occurrence of the best pair in the sequence */
    for (int i = 0; i < s->length - 1; i++) {
        if (s->tokens[i] == best_l && s->tokens[i + 1] == best_r)
            return i;
    }
    return -1;
}

/* Train: learn num_merges merge rules from the text */
static void train_bpe(const unsigned char *text, size_t len, int num_merges) {
    /* Initialise the 2D merge table.
     * Capacity = 256 base bytes + num_merges new tokens we'll learn.
     * We allocate once (naively big) so add_merge is just an array write. */
    int max_id = 256 + num_merges;
    merge_table_init(max_id);
    pair_counts_init(max_id);

    /* Initialize sequence with raw bytes */
    seq_t seq;
    seq_init(&seq, (int)len + 1);
    for (size_t i = 0; i < len; i++) {
        seq_append(&seq, (int)text[i]);
    }

    printf("  Initial tokens:  %d\n", seq.length);
    fflush(stdout);

    for (int m = 0; m < num_merges; m++) {
        if (seq.length < 2) break;

        int left, right;
        int pos = find_most_frequent_pair(&seq, &left, &right);
        if (pos < 0) break;  /* no pairs found */

        add_merge(left, right, m);

        /* Merge ALL occurrences of this pair in the sequence */
        /* After each merge, the sequence shrinks so we re-scan from the start */
        while (1) {
            int found = 0;
            for (int i = 0; i < seq.length - 1; i++) {
                if (seq.tokens[i] == left && seq.tokens[i + 1] == right) {
                    seq_merge_at(&seq, i, next_new_id - 1);
                    found = 1;
                    break;  /* restart scan since array shifted */
                }
            }
            if (!found) break;
        }

        if ((m + 1) % 1000 == 0 || m == 0 || m == num_merges - 1) {
            printf("  Merge %5d/%d: pair (%d,%d) → id=%d, seq_len=%d\r",
                   m + 1, num_merges, left, right, next_new_id - 1, seq.length);
            fflush(stdout);
        }
    }

    printf("\n  Training done. Vocab size = %d (256 base + %d merges)\n",
           256 + num_merges_total, num_merges_total);
    fflush(stdout);

    free(seq.tokens);
}

/* ==================================================================
 *  Naive BPE — ENCODING (Inference)
 *
 *  Given a merge table (already trained), encode new text:
 *    1. Convert text to raw bytes (IDs 0-255)
 *    2. Repeatedly find the lowest-rank (highest-priority) mergeable pair
 *    3. Merge it
 *    4. Repeat until no more mergeable pairs
 *
 *  This is O(N²) in the worst case:
 *    - Each scan is O(N) to find the best pair
 *    - We do up to N merges (each shrinks seq by 1)
 *    - Each merge shifts the array via memmove
 *
 *  Pair lookup is O(1) via the 2D array — no longer O(M).
 * ================================================================== */

static int encode_naive(const unsigned char *text, size_t len,
                         int *output_ids, int max_output) {
    /* Build sequence from raw bytes */
    seq_t seq;
    seq_init(&seq, (int)len + 1);
    for (size_t i = 0; i < len; i++) {
        seq_append(&seq, (int)text[i]);
    }

    while (seq.length >= 2) {
        /* Scan to find the best (lowest-rank) mergeable pair */
        int best_pos = -1;
        int best_rank = 999999999;

        for (int i = 0; i < seq.length - 1; i++) {
            int rank = lookup_rank(seq.tokens[i], seq.tokens[i + 1]);
            if (rank >= 0 && rank < best_rank) {
                best_rank = rank;
                best_pos  = i;
                if (best_rank == 0) break;  /* can't beat rank 0 */
            }
        }

        if (best_pos < 0) break;  /* no mergeable pair exists → done */

        int new_id = lookup_new_id(seq.tokens[best_pos], seq.tokens[best_pos + 1]);
        seq_merge_at(&seq, best_pos, new_id);
    }

    /* Copy results */
    int n = seq.length < max_output ? seq.length : max_output;
    memcpy(output_ids, seq.tokens, n * sizeof(int));
    free(seq.tokens);
    return n;
}

/* ==================================================================
 *  Benchmark driver
 * ================================================================== */

static void run_benchmark(const char *fname, int num_merges, int repeat) {
    /* mmap the file */
    int fd = open(fname, O_RDONLY);
    if (fd < 0) { perror("open"); exit(1); }

    struct stat st;
    fstat(fd, &st);
    size_t file_size = (size_t)st.st_size;

    unsigned char *data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) { perror("mmap"); close(fd); exit(1); }
    close(fd);

    printf("Input:   %s (%.2f MB)\n", fname, file_size / 1e6);
    printf("Merges:  %d\n", num_merges);
    printf("Repeat:  %d\n", repeat);
    printf("Build:   -O0 -g (no optimizations, debug symbols)\n");
    printf("--------------------------------------------------------\n");

    /* ---- TRAINING ---- */
    double t0 = now_sec();
    train_bpe(data, file_size, num_merges);
    double t_train = now_sec() - t0;
    printf("Train time:        %.4f s\n", t_train);

    /* ---- ENCODING ---- */
    int max_output = (int)(file_size / 2 + 1);
    int *output_ids = malloc(max_output * sizeof(int));

    /* Warmup on first 1KB */
    int warmup_len = file_size > 1024 ? 1024 : file_size;
    encode_naive(data, warmup_len, output_ids, max_output);

    double best_time = 1e30;
    double total_time = 0;
    int final_tokens = 0;

    for (int r = 0; r < repeat; r++) {
        double t1 = now_sec();
        int nt = encode_naive(data, file_size, output_ids, max_output);
        double t2 = now_sec();
        double dt = t2 - t1;
        total_time += dt;
        if (dt < best_time) best_time = dt;
        final_tokens = nt;
    }

    double avg_time = total_time / repeat;
    double tokens_per_sec = final_tokens / avg_time;
    double mb_per_sec = (file_size / 1e6) / avg_time;

    printf("\n--- Encoding Results ---\n");
    printf("  Final tokens:   %'d\n", final_tokens);
    printf("  Avg time:       %.4f s\n", avg_time);
    printf("  Best time:      %.4f s\n", best_time);
    printf("  Throughput:     %'.0f tokens/sec\n", tokens_per_sec);
    printf("                  %'.2f MB/sec\n", mb_per_sec);
    printf("  Compression:    %.1f bytes/token\n",
           (double)file_size / final_tokens);

    /* Free everything */
    free(output_ids);
    free_merge_table();
    pair_counts_free();
    munmap(data, file_size);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input.txt> [num_merges=50000] [repeat=1]\n",
                argv[0]);
        return 1;
    }

    const char *fname = argv[1];
    int num_merges = argc > 2 ? atoi(argv[2]) : 50000;
    int repeat     = argc > 3 ? atoi(argv[3]) : 1;

    run_benchmark(fname, num_merges, repeat);
    return 0;
}