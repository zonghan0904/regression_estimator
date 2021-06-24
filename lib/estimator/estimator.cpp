#include <torch/script.h> // One-stop header.
#include "estimator.h"

#include <iostream>
#include <memory>

float pwm[4] = {0};

float* pwm_estimator(float battery, float* f, char* path) {
  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(path);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    exit(-1);
  }

  // std::cout << "ok\n";
  // Create a vector of inputs.
  float data[5] = {0};
  try {
    data[0] = battery;
    data[1] = f[0];
    data[2] = f[1];
    data[3] = f[2];
    data[4] = f[3];
  }
  catch (...) {
      std::cerr << "incorrect input data type\n";
      exit(-1);
  }
  std::vector<torch::jit::IValue> inputs;
  torch::Tensor data_tensor = torch::from_blob(data, {1, 5});
  inputs.push_back(data_tensor);

  // Execute the model and turn its output into a tensor.
  // It should output [[0.642109036446,0.671986341476,0.638709008694,0.654757082462]]
  torch::Tensor output = module.forward(inputs).toTensor();
  // std::cout << output << '\n';

  auto output_a = output.accessor<float, 2>();
  for (int i = 0; i < 4; i++){
      pwm[i] = output_a[0][i];
      // std::cout << "PWM " << i+1 << ": " << pwm[i] << '\n';
  }
  return pwm;
}
