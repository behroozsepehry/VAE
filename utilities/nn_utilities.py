import torch
from torch import nn


def apply_func_to_model_data(model, func, dataloader, device):
    result = 0.
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(dataloader):
            x = x.to(device)
            model_out = model(x)
            result += func(**model_out)
    result /= len(dataloader.sampler)
    return result


def data_parallel_model(model, input, ngpu):
    if 'cuda' in str(input.device) and ngpu > 1:
        output = nn.parallel.data_parallel(model, input, range(ngpu))
    else:
        output = model(input)
    return output


class BsDist(object):
    """Bullshit distribution by Babanezhad--Sepehry
    pdf = (p^x)*((1-p)^(1-x)) / ((1-2*p)/(log(1-p) - log(p)))
    """
    @staticmethod
    def pdf(x, p, eps=1.0e-5):
        # print('x'*10)
        # print(x[0, 0, 20, 10:20])
        # print('p'*10)
        # print(p[0, 0, 20, 10:20])
        p = p.clamp(eps, 1-eps)
        c = ((1 - 2*p).abs() + eps) / (((1-p).log() - (p).log()).abs() + 2*eps)
        f = p.pow(x) * (1-p).pow(1-x)
        d = f / c
        return d

    @staticmethod
    def pdf256(x, p, eps=1.0e-5):
        print('x'*10)
        print(x[0, 0, 20, 10:20])
        print('p'*10)
        print(p[0, 0, 20, 10:20])
        bin_size = 1. / 256.
        # implementation like https://github.com/openai/iaf/blob/master/tf_utils/distributions.py#L28
        x = torch.floor(x / bin_size) * bin_size

        cdf_plus = BsDist.cdf((x + bin_size).clamp(0, 1), p, eps)
        cdf_minus = BsDist.cdf(x, p, eps)

        return cdf_plus - cdf_minus

    @staticmethod
    def cdf(x, p, eps=1.0e-5):
        p = p.clamp(eps, 1-eps)
        cd = ((-p.pow(x) * (1-p).pow(1-x) + (1-p)).abs() + eps*x) / ((1-2*p).abs() + eps)
        return cd


def log_logistic_256(x, mean, logvar, average=False, reduce=True, dim=None):
    # https://github.com/jmtomczak/vae_vampprior/blob/bb6ff3e58036adbea448e82d7bc55593d605b52c/utils/distributions.py#L38
    bin_size = 1. / 256.

    # implementation like https://github.com/openai/iaf/blob/master/tf_utils/distributions.py#L28
    scale = torch.exp(logvar)
    x = (torch.floor(x / bin_size) * bin_size - mean) / scale
    cdf_plus = torch.sigmoid(x + bin_size/scale)
    cdf_minus = torch.sigmoid(x)

    # calculate final log-likelihood for an image
    log_logist_256 = -torch.log(cdf_plus - cdf_minus + 1.e-7)

    if reduce:
        if average:
            return torch.mean(log_logist_256, dim)
        else:
            return torch.sum(log_logist_256, dim)
    else:
        return log_logist_256

