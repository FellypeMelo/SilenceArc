#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct DFState {
    int dummy;
} DFState;

typedef struct {
    uint32_t *array;
    uint32_t length;
} DynArray;

DFState* df_create(const char* path, float atten_lim, const char* log_level) {
    DFState* st = (DFState*)malloc(sizeof(DFState));
    st->dummy = 1;
    return st;
}

uint32_t df_get_frame_length(DFState* st) {
    return 480; // Default for 48kHz 10ms
}

void df_set_atten_lim(DFState* st, float lim_db) {
    // Stub
}

float df_process_frame(DFState* st, float* input, float* output) {
    // Just copy input to output (no suppression)
    memcpy(output, input, 480 * sizeof(float));
    return 20.0f; // Fake SNR
}

void df_free(DFState* model) {
    free(model);
}

// Additional symbols required by headers
char* df_next_log_msg(DFState* st) { return NULL; }
void df_free_log_msg(char* ptr) {}
void df_set_post_filter_beta(DFState* st, float beta) {}
DynArray df_coef_size(const DFState *st) { DynArray a = {0, 0}; return a; }
DynArray df_gain_size(const DFState *st) { DynArray a = {0, 0}; return a; }
float df_process_frame_raw(DFState *st, float *input, float **out_gains_p, float **out_coefs_p) { return 20.0f; }
