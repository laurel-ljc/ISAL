import torch
from isal.interaction import FootInteractionTracker,SelfSupervisedCfg
from isal.learning.auxiliary_buffer import AuxiliaryRolloutBuffer


def test_delayed_finalization_crosses_rollout_without_changing_labels(device):
    x,y=torch.meshgrid(torch.tensor([-.4,.4,1.2]),torch.tensor([-.5,0.,.5]),indexing="ij")
    coords=torch.stack((x.flatten(),y.flatten()),-1).to(device)
    trackers=[FootInteractionTracker(2,(3,3),coords,.02,SelfSupervisedCfg(emit_sample_timing=timing),device)
              for timing in (False,True)]
    def frame(contact):
        force=torch.zeros(2,2,3,device=device);force[...,2]=100. if contact else 0.
        return dict(height_scan=torch.zeros(2,1,3,3,device=device),root_xy=torch.zeros(2,2,device=device),
                    root_yaw=torch.zeros(2,device=device),command=torch.zeros(2,3,device=device),
                    base_ang_vel=torch.zeros(2,3,device=device),projected_gravity=torch.tensor([[0.,0.,-1.]],device=device).repeat(2,1),
                    foot_pos_w=torch.zeros(2,2,3,device=device),foot_vel_w=torch.zeros(2,2,3,device=device),
                    foot_force_w=force,base_roll_pitch=torch.zeros(2,2,device=device),
                    terminated=torch.zeros(2,dtype=torch.bool,device=device),truncated=torch.zeros(2,dtype=torch.bool,device=device))
    buffer=AuxiliaryRolloutBuffer(16,device,123)
    for step in range(1,30):
        old,new=[tracker.update(**frame(step!=2)) for tracker in trackers]
        for key in old:
            assert torch.equal(old[key],new[key]),key
        if step==24:
            assert trackers[1].active.any()
            buffer.clear()  # a learning boundary must not touch live physical events
        buffer.append(new,(step-1)%24+1)
    assert buffer.received==4 and buffer.cross_rollout==4
    assert (buffer.data["snapshot_step"]==2).all()
    assert (buffer.data["finalized_step"]==28).all()
    assert not trackers[1].active.any()
    assert not trackers[1].update(**frame(True))["valid"].any()  # consumed once
    trackers[1].reset(torch.arange(2,device=device))
    assert not trackers[1].pending_snapshot_step.any() and not trackers[1].swing_snapshot_step.any()
