import pytest
import torch


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda":
        assert torch.cuda.is_available(), "CUDA is required for local Stage 4B acceptance."
    return request.param


@pytest.fixture(autouse=True)
def tensor_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)
