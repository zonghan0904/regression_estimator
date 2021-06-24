import torch
import torchvision
from model import RegressionNet

# An instance of your model.
model = RegressionNet(5, 4, 256)
model.load_state_dict(torch.load("./weights/estimator.ckpt"))

# An example input you would normally provide to your model's forward() method.
example = torch.FloatTensor([[11.9030313492,3.00998950005,3.47481918335,3.08211278915,3.16168761253]])

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)

# It should output [[0.642109036446,0.671986341476,0.638709008694,0.654757082462]]
data = torch.FloatTensor([[11.8876552582,3.02776765823,3.37524223328,2.91619372368,3.1193883419]])
output = traced_script_module(data)
print(output)

traced_script_module.save("./weights/traced_regression_net_model.pt")
