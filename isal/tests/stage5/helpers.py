from collections import deque
from types import SimpleNamespace
import torch
from rsl_rl.storage import RolloutStorage
from isal.learning.models import AffordanceActorCritic, AffordanceObservationActorCritic
from isal.learning.ppo_affordance import PPOWithAffordance
from isal.learning.affordance_runner import AffordanceRunner
from stage4b.helpers import inputs,aux_sample


def make_algorithm(device="cpu",mode="predicted",enabled=True,**kwargs):
    obs,common,extra = inputs(grid=(17,9),histories=(3,4),batch=4,device=device)
    obs = obs.to(device)
    if mode == "aux":
        model = AffordanceActorCritic(obs,**common).to(device)
    else:
        model = AffordanceObservationActorCritic(obs,**common,**extra,
                                               affordance_observation={"input_mode":mode}).to(device)
    storage = RolloutStorage("rl",4,4,obs,[23],device)
    algo = PPOWithAffordance(model,storage,device=device,num_learning_epochs=2,num_mini_batches=2,
                             auxiliary_learning=dict(enabled=enabled,min_samples_per_update=2,aux_batch_size=4),**kwargs)
    return algo,obs


def packet(obs,grid=(17,9)):
    # Four environments, two feet, two simultaneous pending slots.
    flat = aux_sample(obs,grid)
    result = {key:value[:,None,None].expand(4,2,2,*value.shape[1:]).clone() for key,value in flat.items()}
    result["valid"] = torch.ones(4,2,2,dtype=torch.bool,device=obs.device)
    result["foot_side"][:,0] = torch.tensor([1.,0.],device=obs.device)
    result["foot_side"][:,1] = torch.tensor([0.,1.],device=obs.device)
    result["snapshot_step"] = torch.full((4,2,2),3,device=obs.device,dtype=torch.long)
    result["finalized_step"] = torch.full((4,2,2),30,device=obs.device,dtype=torch.long)
    result["partial_window"] = torch.zeros(4,2,2,dtype=torch.bool,device=obs.device)
    result["survival"] = torch.ones(4,2,2,device=obs.device)
    return result


def batch_for(algo,obs):
    with torch.no_grad():
        actions = algo.policy.act(obs).clone()
        value = algo.policy.evaluate(obs).clone()
        log = algo.policy.get_actions_log_prob(actions).unsqueeze(-1).clone()
        mu,std = algo.policy.action_mean.clone(),algo.policy.action_std.clone()
    advantage = torch.tensor([[-1.],[.5],[2.],[-.4]],device=obs.device)
    return (obs,actions,value,advantage,value+.3,log,mu,std,(None,None),None)


def fake_runner(algo):
    runner = AffordanceRunner.__new__(AffordanceRunner)
    runner.alg,runner.device = algo,algo.device
    runner.training_definition = algo.training_definition()
    runner.current_learning_iteration = algo.completed_updates
    runner.elapsed_seconds = 12.
    runner.env = SimpleNamespace(reset=lambda:None)
    runner.logger = SimpleNamespace(save_model=lambda *args:None,tot_timesteps=0,tot_time=0,
        cur_reward_sum=torch.zeros(4),cur_episode_length=torch.zeros(4),rewbuffer=deque(),lenbuffer=deque(),ep_extras=[],
        log_dir=None,writer=None)
    return runner
