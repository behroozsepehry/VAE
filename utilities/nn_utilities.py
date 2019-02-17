import torch


def apply_func_to_model_data(model, func, dataloader, device):
    result = 0.
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(dataloader):
            x = x.to(device)
            model_out = model(x)
            result += func(**model_out)
    result /= len(dataloader.sampler)
    return result
