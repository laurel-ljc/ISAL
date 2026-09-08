import importlib.util
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from isal.learning.models import AffordanceObservationActorCritic
from .helpers import inputs


@pytest.mark.parametrize("mode,gate,grid",[("predicted",0.,(17,11)),("predicted",1.,(17,9)),("zero",1.,(19,11))])
def test_complete_export_even_at_zero_gate_and_dynamic_batch(tmp_path,mode,gate,grid):
    obs,common,extra=inputs(grid=grid,histories=(3,4))
    model=AffordanceObservationActorCritic(obs,**common,**extra,
                                         affordance_observation={"input_mode":mode})
    model.update_normalization(obs)
    model.set_affordance_input_gate(gate)
    # Exercise the public checkpoint export loader, including saved gate restoration.
    path=tmp_path/"checkpoint.pt"
    torch.save({"model_state_dict":model.state_dict()},path)
    script_path=Path(__file__).resolve().parents[2]/"scripts/export_actor.py"
    spec=importlib.util.spec_from_file_location("stage4b_export_cli",script_path)
    cli=importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
    directory=cli.export_checkpoint(str(path),str(tmp_path/"export"))
    scripted=torch.jit.load(str(directory/"actor.pt"))
    graph=onnx.load(str(directory/"actor.onnx")); onnx.checker.check_model(graph)
    session=ort.InferenceSession(str(directory/"actor.onnx"),providers=["CPUExecutionProvider"])
    for batch in (1,4):
        data,_,_=inputs(grid=grid,histories=(3,4),batch=batch)
        with torch.inference_mode():
            expected=model.act_inference(data)
            actual=scripted(data["policy"],data["height_scan"])
        torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-5)
        result=session.run(None,{key:data[key].numpy() for key in ("policy","height_scan")})[0]
        np.testing.assert_allclose(result,expected.numpy(),rtol=1e-4,atol=1e-5)
    jit_names=list(scripted.state_dict())
    onnx_names=[item.name for item in graph.graph.initializer]
    for names in (jit_names,onnx_names):
        assert not any("critic" in name for name in names)
        assert any("affordance_head" in name for name in names)==(mode=="predicted")
        assert any("affordance_projection" in name for name in names)==(mode=="predicted")
    if mode=="predicted":
        assert scripted.input_gate.item()==gate
        # Changing a zero-gate TorchScript buffer must activate the retained head.
        if gate==0:
            scripted.input_gate.fill_(1.)
            model.set_affordance_input_gate(1.)
            with torch.inference_mode():
                torch.testing.assert_close(scripted(obs["policy"],obs["height_scan"]),model.act_inference(obs),
                                           rtol=1e-4,atol=1e-5)
    metadata=json.loads((directory/"metadata.json").read_text())
    assert metadata["input_gate"]==gate
    assert metadata["includes_affordance_head"]==(mode=="predicted")
