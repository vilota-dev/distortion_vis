from .kb4 import KB4
from .double_sphere import DoubleSphere
from .radtan8 import RadTan8
from .eucm import EUCM
from .pinhole import Pinhole

def get_camera_from_dict(params, fov = 179):
    
    type = params['camera_type']
    intr = params['intrinsics']

    if type == "kb4":
        return KB4(intr['fx'], intr['fy'], intr['cx'], intr['cy'], 
                   intr['k1'], intr['k2'], intr['k3'], intr['k4'])
    elif type == "ds":
        return DoubleSphere(intr['fx'], intr['fy'], intr['cx'], intr['cy'], 
                    intr['xi'], intr['alpha'])
    elif type == "eucm":
        return EUCM(intr['fx'], intr['fy'], intr['cx'], intr['cy'], 
                    intr['alpha'], intr['beta'], fov)
    elif type == "pinhole":
        return Pinhole(intr['fx'], intr['fy'], intr['cx'], intr['cy'], fov)
    else:
        raise NotImplementedError(f"model {type} not implemented")