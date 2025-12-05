# concrete modeling backend


some notes for inference:
```
import torch
from my_model_module import MyLightningModule # your original LightningModule class
from my_model import MyVanillaModel # your plain torch.nn.Module class

# assuming you saved your model using trainer.save_checkpoint("model.ckpt")
checkpoint_path = "path/to/your/model.ckpt"

# option 1: load the entire LightningModule and then get the model attribute
# this is useful if your LightningModule's __init__ has custom logic
# but you might still need to define MyLightningModule
# model = MyLightningModule.load_from_checkpoint(checkpoint_path=checkpoint_path)
# vanilla_model = model.model # assuming your nn.Module is an attribute named 'model'

# option 2: load only the state_dict and then apply it to your plain nn.Module
# this is often cleaner for deployment as it removes the LightningModule dependency
checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu")) # or 'cuda'

# if your lightning module has a 'model' attribute that is your nn.Module
# you might need to adjust state_dict keys or create your nn.Module separately
# let's assume your MyLightningModule has a 'model' attribute that is a MyVanillaModel
vanilla_model = MyVanillaModel(...) # initialize your plain nn.Module
vanilla_model_state_dict = {}
for k, v in checkpoint["state_dict"].items():
    if k.startswith("model."): # assuming your nn.Module is nested under 'model'
        vanilla_model_state_dict[k[len("model."):]] = v
    else:
        # handle other keys if necessary, or just skip if they're not part of the core model
        pass
vanilla_model.load_state_dict(vanilla_model_state_dict)

vanilla_model.eval() # set to evaluation mode

# now you can use vanilla_model with LitServe
```