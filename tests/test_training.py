"""Training component tests -- model forward pass, losses, V-trace, and short training runs.

Tests the neural network, loss functions, and V-trace computation with
synthetic data. The full training integration test is marked slow.
"""

import os
import tempfile

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Model forward pass
# ---------------------------------------------------------------------------

class TestModelForwardPass:

    @pytest.fixture
    def model(self, obs_shape, num_actions):
        from train_impala import MemoryMazeNet
        m = MemoryMazeNet(obs_shape, num_actions)
        m.train()
        return m

    def test_forward_basic(self, model):
        T, B = 2, 1
        inputs = {
            "frame": torch.randint(0, 256, (T, B, 3, 64, 64), dtype=torch.uint8),
            "last_action": torch.zeros(T, B, dtype=torch.int64),
            "reward": torch.zeros(T, B),
            "done": torch.zeros(T, B, dtype=torch.bool),
        }
        outputs, new_state = model(inputs, model.initial_state(B))
        assert outputs["policy_logits"].shape == (T, B, 6)
        assert outputs["baseline"].shape == (T, B)
        assert outputs["action"].shape == (T, B)

    def test_forward_larger_batch(self, model):
        T, B = 5, 4
        inputs = {
            "frame": torch.randint(0, 256, (T, B, 3, 64, 64), dtype=torch.uint8),
            "last_action": torch.randint(0, 6, (T, B)),
            "reward": torch.randn(T, B),
            "done": torch.zeros(T, B, dtype=torch.bool),
        }
        outputs, _ = model(inputs, model.initial_state(B))
        assert outputs["policy_logits"].shape == (T, B, 6)

    def test_state_changes(self, model):
        T, B = 3, 2
        inputs = {
            "frame": torch.randint(0, 256, (T, B, 3, 64, 64), dtype=torch.uint8),
            "last_action": torch.zeros(T, B, dtype=torch.int64),
            "reward": torch.zeros(T, B),
            "done": torch.zeros(T, B, dtype=torch.bool),
        }
        state = model.initial_state(B)
        _, new_state = model(inputs, state)
        assert not torch.allclose(new_state[0], state[0])

    def test_outputs_finite(self, model):
        T, B = 2, 1
        inputs = {
            "frame": torch.randint(0, 256, (T, B, 3, 64, 64), dtype=torch.uint8),
            "last_action": torch.zeros(T, B, dtype=torch.int64),
            "reward": torch.zeros(T, B),
            "done": torch.zeros(T, B, dtype=torch.bool),
        }
        outputs, _ = model(inputs, model.initial_state(B))
        assert torch.isfinite(outputs["policy_logits"]).all()
        assert torch.isfinite(outputs["baseline"]).all()

    def test_eval_deterministic(self, obs_shape, num_actions):
        from train_impala import MemoryMazeNet
        model = MemoryMazeNet(obs_shape, num_actions)
        model.eval()
        T, B = 1, 1
        inputs = {
            "frame": torch.randint(0, 256, (T, B, 3, 64, 64), dtype=torch.uint8),
            "last_action": torch.zeros(T, B, dtype=torch.int64),
            "reward": torch.zeros(T, B),
            "done": torch.zeros(T, B, dtype=torch.bool),
        }
        state = model.initial_state(B)
        out1, _ = model(inputs, state)
        out2, _ = model(inputs, state)
        assert torch.equal(out1["action"], out2["action"])

    def test_gradient_flows(self, model):
        T, B = 2, 1
        inputs = {
            "frame": torch.randint(0, 256, (T, B, 3, 64, 64), dtype=torch.uint8),
            "last_action": torch.zeros(T, B, dtype=torch.int64),
            "reward": torch.zeros(T, B),
            "done": torch.zeros(T, B, dtype=torch.bool),
        }
        outputs, _ = model(inputs, model.initial_state(B))
        loss = outputs["policy_logits"].sum() + outputs["baseline"].sum()
        loss.backward()
        assert model.block0.conv.weight.grad is not None
        assert model.block0.conv.weight.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# LSTM state reset
# ---------------------------------------------------------------------------

class TestLSTMStateReset:

    def test_done_resets_state(self, obs_shape, num_actions):
        from train_impala import MemoryMazeNet
        model = MemoryMazeNet(obs_shape, num_actions)
        model.eval()
        T, B = 3, 1
        frame = torch.randint(0, 256, (T, B, 3, 64, 64), dtype=torch.uint8)
        base = {"frame": frame, "last_action": torch.zeros(T, B, dtype=torch.int64),
                "reward": torch.zeros(T, B)}
        state = model.initial_state(B)

        _, s_no_done = model({**base, "done": torch.zeros(T, B, dtype=torch.bool)}, state)
        _, s_done = model({**base, "done": torch.tensor([[[False]], [[False]], [[True]]]).squeeze(-1)}, state)
        assert not torch.allclose(s_no_done[0], s_done[0])


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class TestLossFunctions:

    def test_baseline_loss(self):
        from train_impala import compute_baseline_loss
        loss = compute_baseline_loss(torch.tensor([1.0, -2.0, 0.5]))
        expected = 0.5 * (1.0 + 4.0 + 0.25)
        assert abs(loss.item() - expected) < 1e-5

    def test_baseline_loss_zero(self):
        from train_impala import compute_baseline_loss
        assert compute_baseline_loss(torch.zeros(5)).item() == 0.0

    def test_entropy_loss_uniform(self):
        from train_impala import compute_entropy_loss
        loss = compute_entropy_loss(torch.zeros(1, 6))
        expected = np.log(1.0 / 6.0)
        assert abs(loss.item() - expected) < 1e-5

    def test_policy_gradient_loss_zero_advantage(self):
        from train_impala import compute_policy_gradient_loss
        loss = compute_policy_gradient_loss(
            torch.randn(2, 1, 6), torch.zeros(2, 1, dtype=torch.int64), torch.zeros(2, 1))
        assert loss.item() == 0.0


# ---------------------------------------------------------------------------
# V-trace
# ---------------------------------------------------------------------------

class TestVTrace:

    def test_shapes(self):
        from torchbeast.core.vtrace import from_logits
        T, B = 5, 2
        result = from_logits(
            behavior_policy_logits=torch.randn(T, B, 6),
            target_policy_logits=torch.randn(T, B, 6),
            actions=torch.randint(0, 6, (T, B)),
            discounts=torch.ones(T, B) * 0.99,
            rewards=torch.randn(T, B),
            values=torch.randn(T, B),
            bootstrap_value=torch.randn(B),
        )
        assert result.vs.shape == (T, B)
        assert result.pg_advantages.shape == (T, B)

    def test_on_policy_log_rhos(self):
        from torchbeast.core.vtrace import from_logits
        T, B = 3, 1
        logits = torch.randn(T, B, 6)
        result = from_logits(
            behavior_policy_logits=logits, target_policy_logits=logits,
            actions=torch.randint(0, 6, (T, B)),
            discounts=torch.ones(T, B) * 0.99,
            rewards=torch.zeros(T, B),
            values=torch.zeros(T, B),
            bootstrap_value=torch.zeros(B),
        )
        torch.testing.assert_close(result.log_rhos, torch.zeros(T, B), atol=1e-5, rtol=1e-5)

    def test_finite_values(self):
        from torchbeast.core.vtrace import from_logits
        result = from_logits(
            behavior_policy_logits=torch.randn(10, 4, 6),
            target_policy_logits=torch.randn(10, 4, 6),
            actions=torch.randint(0, 6, (10, 4)),
            discounts=torch.ones(10, 4) * 0.99,
            rewards=torch.randn(10, 4),
            values=torch.randn(10, 4),
            bootstrap_value=torch.randn(4),
        )
        assert torch.isfinite(result.vs).all()
        assert torch.isfinite(result.pg_advantages).all()


# ---------------------------------------------------------------------------
# Learn function
# ---------------------------------------------------------------------------

class TestLearnFunction:

    def test_learn_step(self, obs_shape, num_actions):
        from train_impala import MemoryMazeNet, learn
        T, B = 5, 2
        model = MemoryMazeNet(obs_shape, num_actions)
        actor_model = MemoryMazeNet(obs_shape, num_actions)
        actor_model.load_state_dict(model.state_dict())
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)

        batch = {
            "frame": torch.randint(0, 256, (T + 1, B, 3, 64, 64), dtype=torch.uint8),
            "reward": torch.zeros(T + 1, B),
            "done": torch.zeros(T + 1, B, dtype=torch.bool),
            "episode_return": torch.zeros(T + 1, B),
            "episode_step": torch.zeros(T + 1, B, dtype=torch.int32),
            "policy_logits": torch.randn(T + 1, B, num_actions),
            "baseline": torch.zeros(T + 1, B),
            "last_action": torch.randint(0, 6, (T + 1, B)),
            "action": torch.randint(0, 6, (T + 1, B)),
        }
        flags = type("F", (), {
            "reward_clipping": "none", "discounting": 0.99,
            "baseline_cost": 0.5, "entropy_cost": 0.001, "grad_norm_clipping": 40.0,
        })()
        stats = learn(flags, actor_model, model, batch, model.initial_state(B), optimizer, scheduler)
        assert "total_loss" in stats
        assert np.isfinite(stats["total_loss"])

    def test_learn_updates_weights(self, obs_shape, num_actions):
        from train_impala import MemoryMazeNet, learn
        T, B = 3, 1
        model = MemoryMazeNet(obs_shape, num_actions)
        actor_model = MemoryMazeNet(obs_shape, num_actions)
        actor_model.load_state_dict(model.state_dict())
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
        w_before = model.policy.weight.data.clone()

        batch = {
            "frame": torch.randint(0, 256, (T + 1, B, 3, 64, 64), dtype=torch.uint8),
            "reward": torch.ones(T + 1, B),
            "done": torch.zeros(T + 1, B, dtype=torch.bool),
            "episode_return": torch.zeros(T + 1, B),
            "episode_step": torch.zeros(T + 1, B, dtype=torch.int32),
            "policy_logits": torch.randn(T + 1, B, num_actions),
            "baseline": torch.zeros(T + 1, B),
            "last_action": torch.randint(0, 6, (T + 1, B)),
            "action": torch.randint(0, 6, (T + 1, B)),
        }
        flags = type("F", (), {
            "reward_clipping": "none", "discounting": 0.99,
            "baseline_cost": 0.5, "entropy_cost": 0.001, "grad_norm_clipping": 40.0,
        })()
        learn(flags, actor_model, model, batch, model.initial_state(B), optimizer, scheduler)
        assert not torch.equal(model.policy.weight.data, w_before)


# ---------------------------------------------------------------------------
# Integration test (slow)
# ---------------------------------------------------------------------------

class TestTrainingIntegration:

    @pytest.mark.slow
    def test_short_training_run(self):
        from train_impala import parser, train

        with tempfile.TemporaryDirectory() as tmpdir:
            args = parser.parse_args([
                "--num_actors", "1",
                "--total_steps", "200",
                "--batch_size", "1",
                "--unroll_length", "10",
                "--disable_cuda",
                "--disable_checkpoint",
                "--savedir", tmpdir,
                "--seed", "42",
            ])
            train(args)

            xpid_dirs = [d for d in os.listdir(tmpdir) if d.startswith("torchbeast-")]
            assert len(xpid_dirs) == 1
            logs_csv = os.path.join(tmpdir, xpid_dirs[0], "logs.csv")
            assert os.path.exists(logs_csv)
            with open(logs_csv) as f:
                lines = f.readlines()
            assert len(lines) >= 2
