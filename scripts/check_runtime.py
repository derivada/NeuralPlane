import json
import torch
assert torch.__version__.split('+')[0]=='2.7.1'
assert torch.version.cuda=='12.8'
assert torch.cuda.get_device_capability()==(12,0)
x=torch.randn(64,64,device='cuda',requires_grad=True)
x.square().sum().backward()
assert torch.isfinite(x.grad).all()
import tinycudann as tcnn
m=tcnn.Network(3,1,dict(otype='FullyFusedMLP',activation='ReLU',output_activation='None',n_neurons=32,n_hidden_layers=2))
y=m(torch.rand(64,3,device='cuda')); y.sum().backward()
assert torch.isfinite(y).all()
from neuralplane.neuralplane_config import neuralplane_method
print(json.dumps(dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),capability=torch.cuda.get_device_capability())))
