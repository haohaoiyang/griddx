from __future__ import annotations

import platform

import fastapi
import numpy
import pandas
import scipy
import sklearn
import sqlalchemy
import torch


def main() -> int:
    print(f"platform={platform.platform()}")
    print(f"python={platform.python_version()}")
    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"cuda_runtime={torch.version.cuda}")
    print(f"gpu_count={torch.cuda.device_count()}")
    for index in range(torch.cuda.device_count()):
        print(f"gpu_{index}={torch.cuda.get_device_name(index)}")
    print(f"fastapi={fastapi.__version__}")
    print(f"numpy={numpy.__version__}")
    print(f"pandas={pandas.__version__}")
    print(f"scipy={scipy.__version__}")
    print(f"scikit_learn={sklearn.__version__}")
    print(f"sqlalchemy={sqlalchemy.__version__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
