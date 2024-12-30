import torch
from cldm.model import create_model
from cldm.cldm import ControlLDM

model = create_model("./models/cldm_v21.yaml").cpu()
X = torch.rand((64, 64, 3))
y = torch.rand((64, 64, 3))
data = dict(jpg=y, txt=" ", hint=X)
model(X, data)