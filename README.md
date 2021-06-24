# Regression Estimator

## Loss
![](https://user-images.githubusercontent.com/40656204/122343820-b6831080-cf78-11eb-8707-82ea3d54c209.png)

## Compare
![](https://user-images.githubusercontent.com/40656204/122343855-bdaa1e80-cf78-11eb-8fd6-2544c0bacce5.png)

## Usage

### Requirements
* Python3

### Training
* `$ python train.py --valid --verbose`
* the evaluation result will be saved at `./evaluations`
* the model's weight will be saved as `./weights/estimator.ckpt`

### Function Call in C
* convert PyTorch model to Torch script `$ python convert.py`
* the Torch script module will be saved as `./weights/traced_regression_net_model.pt`
* donwload [libtorch](wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip) zip file and unzip it
* move libtorch to `/usr/local/lib`
* compile estimator
    * `$ cd lib/estimator/`
    * `$ mkdir build`
    * `$ cd build`
    * `$ cmake ..`
    * `$ cmake --build . --config Release`
    * `$ mkdir libestimator`
    * `$ mv libestimator.so libestimator`
* move libestimator to `/usr/local/lib`
* `$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/libestimator`
* `$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/libtorch/lib`
* `$ make`
* `$ ./pwm_estimator`
