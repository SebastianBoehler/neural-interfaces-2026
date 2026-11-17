#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
const char* ni26_error(void);
void* ni26_create(const char* directory,const char* device,const char* channels);
void* ni26_clone(void* model);
void ni26_destroy(void* model);
int ni26_reset(void* model);
int ni26_predict(void* model,const float* input,int64_t count,int cuda,int64_t* labels);
#ifdef __cplusplus
}
#endif
