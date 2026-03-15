"""Smoke tests -- quick import and instantiation checks (~2s).

No environment creation, no rendering. These must pass everywhere.
"""

import pytest
import torch


class TestImports:
    """Verify all required packages import without error."""

    def test_import_memory_maze(self):
        import memory_maze

    def test_import_model_class(self):
        from train_impala import MemoryMazeNet
        assert MemoryMazeNet is not None

    def test_import_loss_functions(self):
        from train_impala import (
            compute_baseline_loss,
            compute_entropy_loss,
            compute_policy_gradient_loss,
        )
        assert callable(compute_baseline_loss)

    def test_import_vtrace(self):
        from torchbeast.core import vtrace
        assert hasattr(vtrace, "from_logits")

    def test_import_environment(self):
        from torchbeast.core import environment
        assert hasattr(environment, "Environment")

    def test_import_file_writer(self):
        from torchbeast.core import file_writer
        assert hasattr(file_writer, "FileWriter")

    def test_import_prof(self):
        from torchbeast.core import prof
        assert hasattr(prof, "Timings")


class TestModelInstantiation:
    """Verify MemoryMazeNet can be created with correct structure."""

    def test_instantiate_default(self, obs_shape, num_actions):
        from train_impala import MemoryMazeNet
        model = MemoryMazeNet(obs_shape, num_actions)
        assert model.num_actions == 6

    def test_model_components(self, obs_shape, num_actions):
        from train_impala import MemoryMazeNet
        model = MemoryMazeNet(obs_shape, num_actions)

        assert isinstance(model.core, torch.nn.LSTM)
        assert model.core.input_size == 263  # 256 + 6 + 1
        assert model.core.hidden_size == 256
        assert model.fc.in_features == 2048  # 32 * 8 * 8
        assert model.policy.out_features == 6
        assert model.baseline.out_features == 1

    def test_initial_state_shape(self, obs_shape, num_actions):
        from train_impala import MemoryMazeNet
        model = MemoryMazeNet(obs_shape, num_actions)
        h0, c0 = model.initial_state(batch_size=4)
        assert h0.shape == (1, 4, 256)
        assert c0.shape == (1, 4, 256)
        assert torch.all(h0 == 0)

    def test_parameter_count(self, obs_shape, num_actions):
        from train_impala import MemoryMazeNet
        model = MemoryMazeNet(obs_shape, num_actions)
        total = sum(p.numel() for p in model.parameters())
        assert 500_000 < total < 5_000_000
