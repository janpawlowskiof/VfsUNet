import time

import torch
import torch.backends.cudnn

from src.model.vfs_unet import VfsUNet
from src.model.vrcnn import VrCnn
from src.model.vrcnn_bn import VrCnnBn


def performance_test():
    print("cuda version:", torch.version.cuda)
    print("cudnn version:", torch.backends.cudnn.version())

    results = {
        "VFS-UNet": performance_test_model(VfsUNet(3, 3), channels=3),
        "VRCNN": performance_test_model(VrCnn(), channels=1),
        "VRCNN-BN": performance_test_model(VrCnnBn(), channels=1) / 3
    }

    for name, fps in results.items():
        print(f"{name}: {fps} fps")


def performance_test_model(model, channels, n_items=1000):
    model = model.eval().cuda()

    with torch.no_grad():
        rgb = torch.rand([1, channels, 720, 1280], device="cuda")
        model(rgb)

        start_time = time.time()
        for _ in range(n_items):
            model(rgb)
            torch.cuda.synchronize(device="cuda")
        final_time = time.time() - start_time
    fps = n_items / final_time
    return fps


if __name__ == "__main__":
    performance_test()
