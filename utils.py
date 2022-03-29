import torch

def calc_loss(z, j):
    squared_error = torch.sum(z**2, (1, 2, 3)) / 2
    jacob = torch.sum(j, (1, 2, 3))
    return (squared_error - jacob, squared_error, jacob)

def get_score(z, j):
    # summation for channel dimension
    logpz = torch.sum(0.5 * z**2, 1, keepdim=True)
    jac = torch.sum(j, 1, keepdim=True)
    return logpz - jac