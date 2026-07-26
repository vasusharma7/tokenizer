/**
 * opt_bpe.c
 * Optimized Byte-Pair Encoding with two key improvements over naive_bpe:
 *
 *  Optimization 1 — Arena + Doubly Linked List (no memmove on merge)
 *  Optimization 2 — Min-Heap / Priority Queue for O(log K) pair selection
 *
 *  Build:  cc -O3 -march=native -o opt_bpe opt_bpe.c -lm
 *  Usage:  ./opt_bpe <input.txt> [num_merges] [repeat]
 *
 *  Shares the same interface as naive_bpe so they can be compared directly.
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
 *  O(1) lookup via flat 2D array  (same as naive version — already optimal)
 * ================================================================== */

static int *merge_rank   = NULL;
static int *merge_new_id = NULL;
static int  vocab_capacity = 0;
static int  next_new_id = 256;
static int  num_merges_total = 0;

static void merge_table_init(int max_id) {
    size_t n = (size_t)max_id * max_id;
    merge_rank   = calloc(n, sizeof(int));
    merge_new_id = calloc(n, sizeof(int));
    if (!merge_rank || !merge_new_id) { perror("calloc"); exit(1); }
    for (size_t i = 0; i < n; i++) merge_rank[i] = -1;
    vocab_capacity = max_id;
    next_new_id = 256;
    num_merges_total = 0;
}

static int lookup_rank(int left, int right) {
    if (left >= vocab_capacity || right >= vocab_capacity) return -1;
    return merge_rank[left * vocab_capacity + right];
}

static int lookup_new_id(int left, int right) {
    if (left >= vocab_capacity || right >= vocab_capacity) return -1;
    return merge_new_id[left * vocab_capacity + right];
}

static void add_merge(int left, int right, int rank) {
    int idx = left * vocab_capacity + right;
    merge_rank[idx]   = rank;
    merge_new_id[idx] = next_new_id++;
    num_merges_total++;
}

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
 *  Optimization 1: Arena-Allocated Doubly-Linked List
 *
 *  All nodes live in one contiguous array — zero heap fragmentation,
 *  excellent cache locality.  Merging two adjacent nodes is O(1):
 *  just update a few pointers, no memmove.
 * ================================================================== */

typedef struct {
    uint32_t token_id;   /* the token stored at this node */
    int32_t  prev;       /* index of previous node, -1 = head sentinel */
    int32_t  next;       /* index of next   node, -1 = tail sentinel */
} Node;

static Node  *nodes      = NULL;
static int32_t head_idx  = 0;   /* first active node index */
static int32_t tail_idx  = 0;   /* last  active node index */
static int32_t list_len  = 0;   /* number of active nodes */
static int32_t arena_cap = 0;   /* total capacity of nodes[] */

/* Build a linked list from raw byte text.  Allocates the arena once. */
static void list_from_bytes(const unsigned char *text, size_t len) {
    arena_cap = (int32_t)(len + num_merges_total + 1024);  /* room for merge tokens */
    nodes = calloc(arena_cap, sizeof(Node));
    if (!nodes) { perror("calloc"); exit(1); }

    for (size_t i = 0; i < len; i++) {
        nodes[i].token_id = (uint32_t)text[i];
        nodes[i].prev     = (int32_t)i - 1;
        nodes[i].next     = (int32_t)i + 1;
    }
    nodes[0].prev      = -1;
    nodes[len - 1].next = -1;
    head_idx = 0;
    tail_idx = (int32_t)len - 1;
    list_len = (int32_t)len;
}

/* Merge the pair at (pos, pos.next) into a single node.
 *  - left node gets the new_id
 *  - right node is "deleted" (skipped over)
 *  - Updates only 4 pointers — O(1), no memmove! */
static void list_merge_at(int32_t pos, uint32_t new_id) {
    int32_t right = nodes[pos].next;
    if (right < 0) return;  /* safety: pos was already the tail */

    nodes[pos].token_id = new_id;    /* left node takes the new token */
    nodes[pos].next     = nodes[right].next;  /* skip over right node */

    if (nodes[right].next >= 0)
        nodes[nodes[right].next].prev = pos;   /* update backward link */
    else
        tail_idx = pos;  /* pos is now the tail */

    /* Mark right node as deleted by pointing its prev to itself.
     * Helps heap detect stale entries. */
    nodes[right].prev = right;
    nodes[right].next = right;

    list_len--;
}

/* Walk the list and fill output_ids[]. Returns the number of tokens. */
static int list_to_ids(int *output_ids, int max_output) {
    int count = 0;
    for (int32_t i = head_idx; i >= 0 && count < max_output; i = nodes[i].next) {
        output_ids[count++] = (int)nodes[i].token_id;
    }
    return count;
}

static void list_free(void) {
    free(nodes);
    nodes = NULL;
    arena_cap = 0;
    list_len = 0;
}

/* ==================================================================
 *  Pair counting for training (uses flat 2D array — same as naive)
 *
 *  Still O(N) to count + O(vocab²) to find max.  The speedup during
 *  training comes from O(1) merges (no memmove) instead.
 * ================================================================== */

static int *pair_counts = NULL;
static int  pair_counts_cap = 0;

static void pair_counts_init(int max_id) {
    size_t n = (size_t)max_id * max_id;
    pair_counts = calloc(n, sizeof(int));
    if (!pair_counts) { perror("calloc"); exit(1); }
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

/* Count pairs and find the most frequent one.
 * Walks the linked list (not an array) — O(N) count + O(vocab²) max scan. */
static int find_most_frequent_pair(int32_t *out_pos, uint32_t *out_left, uint32_t *out_right) {
    pair_counts_reset();
    int cap = pair_counts_cap;

    /* Walk the linked list counting pairs — O(N) */
    for (int32_t i = head_idx; i >= 0; i = nodes[i].next) {
        int32_t j = nodes[i].next;
        if (j < 0) break;
        uint32_t l = nodes[i].token_id;
        uint32_t r = nodes[j].token_id;
        if (l < (uint32_t)cap && r < (uint32_t)cap)
            pair_counts[l * cap + r]++;
    }

    /* Find highest count — O(vocab²) */
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

    if (best_l < 0) return 0;  /* no pairs */

    *out_left  = (uint32_t)best_l;
    *out_right = (uint32_t)best_r;

    /* Find first occurrence */
    for (int32_t i = head_idx; i >= 0; i = nodes[i].next) {
        int32_t j = nodes[i].next;
        if (j < 0) break;
        if (nodes[i].token_id == (uint32_t)best_l && nodes[j].token_id == (uint32_t)best_r) {
            *out_pos = i;
            return 1;
        }
    }
    return 0;
}

/* ==================================================================
 *  TRAINING (uses arena list for O(1) merges)
 * ================================================================== */

static void train_bpe(const unsigned char *text, size_t len, int num_merges) {
    int max_id = 256 + num_merges;
    merge_table_init(max_id);
    pair_counts_init(max_id);
    list_from_bytes(text, len);

    printf("  Initial tokens:  %d\n", list_len);
    fflush(stdout);

    for (int m = 0; m < num_merges; m++) {
        if (list_len < 2) break;

        int32_t pos;
        uint32_t left, right;
        if (!find_most_frequent_pair(&pos, &left, &right))
            break;

        add_merge((int)left, (int)right, m);
        uint32_t new_id = (uint32_t)(next_new_id - 1);

        /* Merge ALL occurrences of (left, right) */
        /* Walk the list, merge on match, then continue from same position
         * (since the list shrinks but the current node is still valid) */
        while (1) {
            int found = 0;
            for (int32_t i = head_idx; i >= 0; ) {
                int32_t j = nodes[i].next;
                if (j < 0) break;
                if (nodes[i].token_id == left && nodes[j].token_id == right) {
                    list_merge_at(i, new_id);
                    found = 1;
                    /* Don't advance i — after merge, i now has new_id
                     * and i.next is the node after the old j. Continue
                     * scanning from here. */
                    continue;
                }
                i = nodes[i].next;
            }
            if (!found) break;
        }

        if ((m + 1) % 1000 == 0 || m == 0 || m == num_merges - 1) {
            printf("  Merge %5d/%d: pair (%u,%u) → id=%u, seq_len=%d\r",
                   m + 1, num_merges, left, right, new_id, list_len);
            fflush(stdout);
        }
    }

    printf("\n  Training done. Vocab size = %d (256 base + %d merges)\n",
           256 + num_merges_total, num_merges_total);
    fflush(stdout);
}

/* ==================================================================
 *  Optimization 2: Min-Heap / Priority Queue
 *
 *  Stores (rank, position_of_left_node) pairs.
 *  Smallest rank = highest priority.
 *  Pop is O(log K), push is O(log K).
 * ================================================================== */

typedef struct {
    int     rank;
    int32_t pos;   /* index of the LEFT node in the pair */
} HeapEntry;

static HeapEntry *heap = NULL;
static int        heap_size = 0;
static int        heap_cap = 0;

static void heap_init(int cap_hint) {
    heap_cap = cap_hint > 0 ? cap_hint : 65536;
    heap = malloc(heap_cap * sizeof(HeapEntry));
    if (!heap) { perror("malloc"); exit(1); }
    heap_size = 0;
}

static void heap_swap(int i, int j) {
    HeapEntry t = heap[i];
    heap[i] = heap[j];
    heap[j] = t;
}

static void heap_push(int rank, int32_t pos) {
    if (heap_size >= heap_cap) {
        heap_cap *= 2;
        heap = realloc(heap, heap_cap * sizeof(HeapEntry));
        if (!heap) { perror("realloc"); exit(1); }
    }
    heap[heap_size].rank = rank;
    heap[heap_size].pos  = pos;
    int i = heap_size;
    heap_size++;

    /* Bubble up */
    while (i > 0) {
        int p = (i - 1) / 2;
        if (heap[p].rank <= heap[i].rank) break;
        heap_swap(p, i);
        i = p;
    }
}

static int heap_pop(int *out_rank, int32_t *out_pos) {
    if (heap_size <= 0) return 0;
    *out_rank = heap[0].rank;
    *out_pos  = heap[0].pos;

    heap_size--;
    heap[0] = heap[heap_size];

    /* Bubble down */
    int i = 0;
    while (1) {
        int smallest = i;
        int l = 2 * i + 1;
        int r = 2 * i + 2;
        if (l < heap_size && heap[l].rank < heap[smallest].rank) smallest = l;
        if (r < heap_size && heap[r].rank < heap[smallest].rank) smallest = r;
        if (smallest == i) break;
        heap_swap(i, smallest);
        i = smallest;
    }
    return 1;
}

static void heap_free(void) {
    free(heap);
    heap = NULL;
    heap_size = 0;
    heap_cap = 0;
}

/* Validate that a heap entry (pos, rank) is still valid:
 *  - pos points to a non-deleted node (node.prev != node)
 *  - nodes[pos].next is valid
 *  - the pair (nodes[pos], nodes[next]) has rank == entry's rank
 *  - the pair still exists in the merge table */
static int heap_entry_valid(int32_t pos, int expected_rank) {
    if (pos < 0 || pos >= arena_cap) return 0;
    /* Deleted nodes have prev == next == self */
    if (nodes[pos].prev == pos && nodes[pos].next == pos) return 0;
    int32_t nxt = nodes[pos].next;
    if (nxt < 0 || nxt >= arena_cap) return 0;
    if (nodes[nxt].prev != pos) return 0;  /* not actually adjacent */

    int rank = lookup_rank((int)nodes[pos].token_id, (int)nodes[nxt].token_id);
    return rank == expected_rank;
}

/* ==================================================================
 *  ENCODING (uses arena list + min-heap)
 *
 *  Instead of scanning the full sequence each iteration, we maintain
 *  a min-heap of all current adjacent pairs.  When we merge, only
 *  2 neighboring pairs are affected — we push those as new entries.
 * ================================================================== */

static int encode_optimized(const unsigned char *text, size_t len,
                             int *output_ids, int max_output) {
    /* Build linked list */
    list_from_bytes(text, len);

    /* Determine how many merge rules we have — scan to find the max rank */
    int max_rank = -1;
    for (int32_t i = head_idx; i >= 0; i = nodes[i].next) {
        int32_t j = nodes[i].next;
        if (j < 0) break;
        int r = lookup_rank((int)nodes[i].token_id, (int)nodes[j].token_id);
        if (r > max_rank) max_rank = r;
    }

    /* Initialise heap with capacity for all initial pairs + 2 per merge */
    heap_init((int)(len + num_merges_total * 2));

    /* Push all initial adjacent pairs */
    for (int32_t i = head_idx; i >= 0; i = nodes[i].next) {
        int32_t j = nodes[i].next;
        if (j < 0) break;
        int rank = lookup_rank((int)nodes[i].token_id, (int)nodes[j].token_id);
        if (rank >= 0)
            heap_push(rank, i);
    }

    /* Pop-and-merge loop */
    while (heap_size > 0) {
        int rank;
        int32_t pos;
        if (!heap_pop(&rank, &pos)) break;

        /* Skip stale entries */
        if (!heap_entry_valid(pos, rank)) continue;

        int32_t right = nodes[pos].next;
        uint32_t new_id = (uint32_t)lookup_new_id((int)nodes[pos].token_id,
                                                   (int)nodes[right].token_id);

        /* Record the left neighbor BEFORE the merge (may become stale) */
        int32_t left_neighbor = nodes[pos].prev;

        /* O(1) merge — 4 pointer updates */
        list_merge_at(pos, new_id);

        if (list_len < 2) break;

        /* After merging pos & right, the pair (left_neighbor, pos) changed.
         * Push it if it's mergeable. */
        if (left_neighbor >= 0) {
            int r = lookup_rank((int)nodes[left_neighbor].token_id, (int)nodes[pos].token_id);
            if (r >= 0) heap_push(r, left_neighbor);
        }

        /* The pair (pos, pos.next) is new — push it if mergeable */
        int32_t new_right = nodes[pos].next;
        if (new_right >= 0) {
            int r = lookup_rank((int)nodes[pos].token_id, (int)nodes[new_right].token_id);
            if (r >= 0) heap_push(r, pos);
        }
    }

    /* Collect results */
    int n = list_to_ids(output_ids, max_output);
    list_free();
    heap_free();
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
    printf("Build:   -O3 -march=native (optimized)\n");
    printf("Opts:    Arena list + Min-heap\n");
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
    encode_optimized(data, warmup_len, output_ids, max_output);

    double best_time = 1e30;
    double total_time = 0;
    int final_tokens = 0;

    for (int r = 0; r < repeat; r++) {
        double t1 = now_sec();
        int nt = encode_optimized(data, file_size, output_ids, max_output);
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
    if (nodes) list_free();
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