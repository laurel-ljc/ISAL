"""Explicit model initialization and reproducible sparse training state."""
import hashlib
import random
import numpy as np
import torch


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


def model_digest(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def extra_state(runner):
    raw = getattr(runner.env, 'unwrapped', None)
    if raw is not None and hasattr(raw, 'course_state_dict'):
        return dict(course_state=raw.course_state_dict(), rng_state=rng_state())
    if raw is None or not hasattr(raw, 'sparse_state_dict'):
        return {}
    return dict(sparse_state=raw.sparse_state_dict(), rng_state=rng_state())


def restore_sparse(runner, checkpoint):
    raw = getattr(runner.env, 'unwrapped', None)
    if raw is not None and hasattr(raw, 'course_state_dict'):
        raw.load_course_state_dict(checkpoint['course_state'])
        restore_rng(checkpoint['rng_state'])
        runner.env.reset()
        return
    if raw is None or not hasattr(raw, 'sparse_state_dict'):
        return
    if 'sparse_state' not in checkpoint:
        raise ValueError('Old checkpoints require --warm-start for a sparse task')
    raw.load_sparse_state_dict(checkpoint['sparse_state'])
    restore_rng(checkpoint['rng_state'])
    runner.env.reset()


def warm_start(runner, path, std=.30):
    checkpoint = torch.load(path, map_location=runner.device, weights_only=False)
    runner.alg.policy.load_state_dict(checkpoint['model_state_dict'], strict=True)
    runner.alg.optimizer.state.clear()
    runner.alg.learning_rate = runner.cfg['algorithm']['learning_rate']
    for group in runner.alg.optimizer.param_groups:
        group['lr'] = runner.alg.learning_rate
    with torch.no_grad():
        policy = runner.alg.policy
        if policy.noise_std_type == 'scalar':
            policy.std.fill_(std)
        else:
            policy.log_std.fill_(float(np.log(std)))
    runner.current_learning_iteration = 0
    if hasattr(runner, 'replay'):
        runner.replay.valid.zero_()
        runner.replay.next = 0
        runner.supervised_optimizer.state.clear()
        runner.total_samples = runner.supervised_updates = runner.iteration = 0
        runner.alg.policy.affordance_alpha.zero_()
        collector = runner.env.unwrapped.collector
        collector.reset(torch.arange(runner.env.num_envs, device=runner.env.device))
        collector.ready.clear()
        collector.stats = dict.fromkeys(collector.stats, 0)
    return checkpoint.get('infos')


def advance(runner, path, report):
    from isal2.deprecated_tasks.sparse.evaluation import check_advance
    checkpoint = torch.load(path, map_location=runner.device, weights_only=False)
    raw = runner.env.unwrapped
    check_advance(checkpoint, raw.cfg.sparse.signature(), report, raw.cfg.terrain_preset)
    runner.alg.policy.load_state_dict(checkpoint['model_state_dict'], strict=True)
    runner.alg.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    runner.alg.learning_rate = runner.alg.optimizer.param_groups[0]['lr']
    runner.current_learning_iteration = checkpoint['iter']
    old = checkpoint['sparse_state']['curriculum']
    current = raw.sparse_curriculum
    current.flat_baseline = old.get('flat_baseline')
    current.entry_report = report
    rescue = checkpoint['sparse_state']['terrain'] == 'sparse_rescue'
    for kind in current.unlocked:
        current.unlocked[kind] = 0 if rescue else min(old['unlocked'].get(kind, 0), current.max_level)
    old_ability = {}
    for kind, level in zip(old['kinds'], old['ability'].tolist()):
        old_ability[kind] = max(old_ability.get(kind, 0), level)
    command_changed = checkpoint['sparse_state']['signature']['command_stage'] != raw.cfg.sparse.command_stage
    for i, kind in enumerate(current.kinds):
        current.ability[i] = min(old_ability.get(kind, 0), 2 if command_changed else raw.cfg.sparse.target_level,
                                 current.unlocked[kind])
    current.assigned.copy_(current.ability)
    if hasattr(runner, 'replay'):
        state = checkpoint.get('affordance_state')
        if state is None or state['config'] != runner.aux_cfg:
            raise ValueError('Phase migration requires matching Affordance training configuration')
        runner.supervised_optimizer.load_state_dict(state['optimizer'])
        runner.total_samples, runner.supervised_updates = state['total_samples'], state['supervised_updates']
        runner.iteration = runner.current_learning_iteration
        runner.replay.valid.zero_()
        runner.replay.next = 0
    raw.sparse_phase_updates = 0
    t = raw.scene.terrain
    t.terrain_levels[:] = current.assigned
    t.env_origins[:] = t.terrain_origins[t.terrain_levels, t.terrain_types]
    restore_rng(checkpoint['rng_state'])
    runner.env.reset()
