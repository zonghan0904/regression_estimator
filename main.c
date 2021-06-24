# include <stdio.h>
# include <stdlib.h>
# include <string.h>
# include "estimator.h"

int main(){
    float battery = 11.8876552582;
    float f[] = {3.02776765823,3.37524223328,2.91619372368,3.1193883419};
    char* path = "./weights/traced_regression_net_model.pt";

    float pwm[4] = {0};
    memcpy(pwm, pwm_estimator(battery, f, path), sizeof(float) * 4);
    for (int i = 0; i < 4; i++){
	printf("PWM %d: %.3f\n", i+1, pwm[i]);
    }

    return 0;
}
