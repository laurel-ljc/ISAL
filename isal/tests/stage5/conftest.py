import pytest
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.runners import OnPolicyRunner
from isal.learning.ppo_affordance import PPOWithAffordance
from isal.learning.affordance_runner import AffordanceRunner


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda":
        assert torch.cuda.is_available()
    return request.param


@pytest.fixture(autouse=True)
def no_learning(monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError("Stage 5 local tests prohibit learn/update/optimizer step.")
    for cls,name in ((PPO,"update"),(PPOWithAffordance,"update"),(OnPolicyRunner,"learn"),(AffordanceRunner,"learn"),
                     (torch.optim.Adam,"step"),(torch.optim.AdamW,"step"),(torch.optim.SGD,"step")):
        monkeypatch.setattr(cls,name,forbidden)
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)
