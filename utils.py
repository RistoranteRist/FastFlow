import torch
import torchvision.io as tvio
import torchvision.transforms as T

def calc_loss(z, j):
    squared_error = torch.sum(z**2, (1, 2, 3)) / 2
    jacob = torch.sum(j, (1, 2, 3))
    return (squared_error - jacob, squared_error, jacob)

def get_score(z, j):
    # summation for channel dimension
    logpz = torch.sum(0.5 * z**2, 1, keepdim=True)
    jac = torch.sum(j, 1, keepdim=True)
    return logpz - jac

def load_image(image_size, path):
    ext = path.split(".")[-1]
    assert ext == "png", "extension {} is not supported.".format(ext)

    trans = T.Compose([
        T.Resize(image_size),
        T.Lambda(lambda im: im / 255.0)
    ])
    image = trans(tvio.read_image(p)).float()
    return image