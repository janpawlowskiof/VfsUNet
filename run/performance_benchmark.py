import time
from pprint import pprint

import torch
import torch.backends.cudnn

from src.model.vfs_unet import VfsUNet
from src.model.vrcnn import VrCnn
from src.model.vrcnn_bn import VrCnnBn


def optimize_model(model):
    model = model.eval().cuda()
    # model = torch.jit.script(model)
    return model


def performance_test():
    print(torch.version.cuda)
    print(torch.backends.cudnn.version())

    results = {
        "VFS-UNet": performance_test_model(VfsUNet(3, 3), channels=3),
        "VRCNN": performance_test_model(VrCnn(), channels=1),
        "VRCNN-BN": performance_test_model(VrCnnBn(), channels=1) / 3
    }

    pprint(results)
    return results


def performance_test_model(model, channels, n_items=100):
    model = optimize_model(model)
    rgb = torch.rand([1, channels, 720, 1280], device="cuda")
    model(rgb)

    with torch.no_grad():
        start_time = time.time()
        for _ in range(n_items):
            model(rgb)
        final_time = time.time() - start_time
    fps = n_items / final_time
    return fps


if __name__ == "__main__":
    performance_test()
