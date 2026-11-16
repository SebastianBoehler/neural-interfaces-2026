#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
const char *n5_error(void);
void *n5_create(int arm, int device, int64_t seed);
void *n5_clone(void *handle);
void n5_destroy(void *handle);
int n5_features(const float *raw, int64_t n, float *out);
int n5_set_norm(void *handle, const float *mu, const float *sd);
int64_t n5_param_count(void *handle);
int n5_get_params(void *handle, float *out);
int n5_set_params(void *handle, const float *values, int64_t count);
int n5_head_forward(void *handle, const float *features, int64_t n, float *logits);
int n5_head_grads(void *handle, const float *features, const int64_t *labels,
                  const float *weights, const float *mask, int64_t n, float *loss,
                  float *input_grad, float *param_grad);
int n5_adam_step(void *handle, const float *features, const int64_t *labels,
                 const float *weights, const float *mask, int64_t n, double lr, double wd);
int n5_predict(void *handle, const float *raw, int64_t n, float *logits);
int n5_fit(void *handle, const float *raw, const int64_t *labels, const int32_t *person,
           int64_t n, int32_t heldout, int epochs, int consistency);
int n5_reset(void *handle);
int n5_step(void *handle, const float *raw_one, int update, float *logits);
int n5_status(void *handle, int64_t *windows, int64_t *updates, int64_t *prefix_count,
              double *max_grad, double *delta);
int n5_fit_status(void *handle, int64_t *steps, double *consistency_sum,
                  double *projection_grad_max, uint64_t *shuffle_hash);
int n5_consistency_grad(void *handle, double *out);
int n5_save(void *handle, const char *path);
int n5_load(void *handle, const char *path);
#ifdef __cplusplus
}
#endif
