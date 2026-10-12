"""Smoke tests for continual_avatar logic (no gsplat / torch CUDA needed).

Verifies:
  - EWCPrior: snapshot, penalty, importance update.
  - ReplayBuffer: azimuth binning + capacity enforcement.
  - VisibilityAccumulator: counts, per-gaussian threshold, resize.
  - ContinualRefiner: construction + config (refine loop needs gsplat, skipped).
"""
import torch

from continual_avatar.ewc import EWCPrior, PARAM_NAMES
from continual_avatar.replay import ReplayBuffer
from continual_avatar.reactivation import VisibilityAccumulator


class _FakeModel:
    """Minimal stand-in for SkinnedGaussians parameter access."""
    def __init__(self, n=10, device="cpu"):
        self._means = torch.randn(n, 3, device=device)
        self._scales = torch.zeros(n, 3, device=device)
        self._quats = torch.zeros(n, 4, device=device)
        self._opacities = torch.zeros(n, device=device)
        self._sh0 = torch.zeros(n, 1, 3, device=device)
        self._skin_weights = None
        self._weights_optimizable = False
        self.n_gaussians = n


def test_ewc():
    m = _FakeModel(8)
    ewc = EWCPrior(gamma=0.9, device="cpu")
    # before any update, penalty is zero.
    assert ewc.penalty(m).item() == 0.0
    # simulate squared grads and update.
    grads = [("_means", torch.ones(8, 3)), ("_scales", torch.ones(8, 3))]
    ewc.update(m, grads, scale=2.0)
    # after update, importance for _means should be 0.5 (=1/2).
    assert torch.allclose(ewc.importance["_means"], torch.full((8, 3), 0.5))
    # penalty should be > 0 if we move the means.
    m._means.add_(0.1)
    assert ewc.penalty(m).item() > 0.0
    # second update with shape change (densification) resets importance.
    m2 = _FakeModel(12)
    ewc.update(m2, [("_means", torch.ones(12, 3))], scale=1.0)
    assert ewc.importance["_means"].shape == (12, 3)
    print("  EWC: OK")


def test_replay():
    rb = ReplayBuffer(capacity=8, n_bins=4)
    assert len(rb) == 0

    class F:
        def __init__(self, yaw):
            self.global_orient = torch.tensor([yaw, 0.0, 0.0])
            self.body_pose = torch.randn(69)

    # add frames with varied azimuth.
    for i in range(40):
        rb.add(F(i * 0.3 - 2.0))
    # capacity is 8 over 4 bins => 2 per bin.
    assert len(rb) <= 8, f"buffer exceeded capacity: {len(rb)}"
    s = rb.sample(3)
    assert len(s) <= 3
    print(f"  Replay: OK (buffer size {len(rb)}, sampled {len(s)})")


def test_reactivation():
    vis = VisibilityAccumulator(n_gaussians=5, rho=0.5, base_grad=2e-4, device="cpu")
    # before any visibility, threshold equals base.
    thr = vis.per_gaussian_threshold()
    assert torch.allclose(thr, torch.full((5,), 2e-4))
    # mark some gaussians visible.
    vis.counts = torch.tensor([10.0, 0.0, 5.0, 0.0, 0.0])
    thr = vis.per_gaussian_threshold()
    # gaussian 0 (max visibility) has highest threshold.
    assert thr[0] > thr[1] and thr[0] > thr[2]
    # gaussian 1 (unseen) has base threshold (float32 tolerance).
    assert abs(thr[1].item() - 2e-4) < 1e-9
    # grad_scale_factors: base_grad / threshold, so unseen gaussians get 1.0.
    sf = vis.grad_scale_factors()
    assert abs(sf[1].item() - 1.0) < 1e-9
    assert sf[0] < sf[1]  # most-visible gaussian gets lowest scale factor.
    # resize after densification.
    vis.resize(7)
    assert vis.counts.shape[0] == 7
    print("  Reactivation: OK")


def test_refiner_construct():
    # only test construction; the refinement loop needs gsplat.
    from continual_avatar.continual import ContinualRefiner, RefinerConfig

    class FakeSMPL:
        pass

    m = _FakeModel(6)
    cfg = RefinerConfig(device="cpu")
    r = ContinualRefiner(m, FakeSMPL(), cfg)
    assert r.ewc is not None and r.replay is not None and r.vis is not None
    assert r.vis.counts.shape[0] == 6
    print("  Refiner construct: OK")


if __name__ == "__main__":
    print("Running continual_avatar smoke tests...")
    test_ewc()
    test_replay()
    test_reactivation()
    test_refiner_construct()
    print("All smoke tests passed.")
