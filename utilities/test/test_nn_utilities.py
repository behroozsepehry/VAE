import pytest
import torch

from utilities import nn_utilities


@pytest.mark.parametrize("x, p, pdf, cdf",
                         [
                             (
                                 torch.Tensor([0.5, 0.3]), torch.Tensor([0.5, 0.6]), torch.Tensor([1, 0.915822]), torch.Tensor([0.5, 0.258694]),
                             ),
                         ]
                         )
def test_bs_dist(x, p, pdf, cdf):
    bs_dist = nn_utilities.BsDist()
    pdf_ = bs_dist.pdf(x, p)
    cdf_ = bs_dist.cdf(x, p)
    print(pdf_, pdf)
    print(cdf_, cdf)
    assert torch.allclose(pdf_, pdf, rtol=1.0e-2)
    assert torch.allclose(cdf_, cdf, rtol=1.0e-2)


if __name__ == '__main__':
    test_bs_dist(torch.Tensor([0.5, 0.3]), torch.Tensor([0.5, 0.6]), torch.Tensor([1, 0.915822]), torch.Tensor([0.5, 0.258694]),)