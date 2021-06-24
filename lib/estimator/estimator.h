#ifndef __ESTIMATOR__H__
#define __ESTIMATOR__H__

#ifdef __cplusplus
extern "C"
{
#endif

extern float pwm[4];
extern float* pwm_estimator(float battery, float *f, char* path);

#ifdef __cplusplus
}
#endif

#endif
