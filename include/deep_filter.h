#ifndef DEEP_FILTER_H
#define DEEP_FILTER_H

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#ifdef _WIN32
  #ifdef DF_EXPORTS
    #define DF_API __declspec(dllexport)
  #else
    #define DF_API __declspec(dllimport)
  #endif
#else
  #define DF_API
#endif

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef struct DFState DFState;

typedef struct {
    uint32_t *array;
    uint32_t length;
} DynArray;

DF_API DFState *df_create(const char *path, float atten_lim, const char *log_level);

DF_API size_t df_get_frame_length(DFState *st);

DF_API char *df_next_log_msg(DFState *st);

DF_API void df_free_log_msg(char *ptr);

DF_API void df_set_atten_lim(DFState *st, float lim_db);

DF_API void df_set_post_filter_beta(DFState *st, float beta);

DF_API float df_process_frame(DFState *st, float *input, float *output);

DF_API float df_process_frame_raw(DFState *st,
                           float *input,
                           float **out_gains_p,
                           float **out_coefs_p);

DF_API DynArray df_coef_size(const DFState *st);

DF_API DynArray df_gain_size(const DFState *st);

DF_API void df_free(DFState *model);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // DEEP_FILTER_H
