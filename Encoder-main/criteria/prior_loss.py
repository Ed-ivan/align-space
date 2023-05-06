import torch
def compute_mvg(latents):
    loss=0.0
    for index in range(latents.shape[0]):
        loss+=latents[index].matmul(latents[index].T)
    return loss
