import json

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from conftest import model_inputs


@pytest.mark.parametrize("grid,histories,aux", [((17, 11), (10, 10), False),
                                              ((17, 9), (3, 4), True),
                                              ((19, 11), (3, 4), True)])
def test_export_torchscript_onnx_dynamic_batch_and_no_head(tmp_path, grid, histories, aux):
    cls, obs, kw = model_inputs(grid, histories=histories)
    model = cls(obs, affordance_head_enabled=aux, **kw)
    model.update_normalization(obs)
    model.eval()
    directory = model.export_actor(tmp_path)
    scripted = torch.jit.load(str(directory / "actor.pt"))
    graph = onnx.load(str(directory / "actor.onnx"))
    onnx.checker.check_model(graph)
    session = ort.InferenceSession(str(directory / "actor.onnx"), providers=["CPUExecutionProvider"])
    for batch in (1, 4):
        _, data, _ = model_inputs(grid, histories=histories, batch=batch)
        with torch.inference_mode():
            expected = model.act_inference(data)
            actual = scripted(data["policy"], data["height_scan"])
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        result = session.run(None, {"policy": data["policy"].numpy(), "height_scan": data["height_scan"].numpy()})[0]
        np.testing.assert_allclose(result, expected.numpy(), rtol=1e-4, atol=1e-5)
    for name in [*scripted.state_dict(), *(i.name for i in graph.graph.initializer)]:
        assert "affordance" not in name and "critic" not in name, name
    meta = json.loads((directory / "metadata.json").read_text())
    assert meta["includes_affordance_head"] is False
    assert meta["layout"]["grid_shape"] == list(grid)
    assert not model.training
