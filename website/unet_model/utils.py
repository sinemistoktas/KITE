import os
import sys
import pathlib

current_dir = pathlib.Path(__file__).parent.absolute()
project_root = current_dir.parent.parent
sys.path.append(str(project_root / "unet" / "src"))

# Now we can import from the UNet source
from pytorch_unet import UNet

# Make the model constructor available
def create_unet(num_class=10, feature_map_size=16, num_task=1):
    return UNet(num_class, f_size=feature_map_size, task_no=num_task)