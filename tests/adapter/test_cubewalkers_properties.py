import unittest
import numpy as np
import cupy as cp
from hypothesis import given, strategies as st, settings

from sdmarkov.adapter.cubewalkers import CWSingleStep, simulate_one_step_aligned_states

# ---------------- Strategies ----------------
@st.composite
def simple_boolean_rules(draw, min_nodes=2, max_nodes=4):
    N = draw(st.integers(min_nodes, max_nodes))
    node_names = [f"X{i}" for i in range(N)]
    rules = []
    for node in node_names:
        op = draw(st.sampled_from(["", "!"]))
        rules.append(f"{node}* = {op}{node}")
    rules_str = "\n".join(rules)
    return N, rules_str, node_names

@st.composite
def identity_networks(draw, min_nodes=1, max_nodes=5):
    N = draw(st.integers(min_nodes, max_nodes))
    nodes = [f"X{i}" for i in range(N)]
    rules_str = "\n".join(f"{node}* = {node}" for node in nodes)
    return N, nodes, rules_str

# ---------------- Tests for CWSingleStep ----------------
class TestCWSingleStepProperties(unittest.TestCase):
    @settings(max_examples=10, deadline=None)
    @given(simple_boolean_rules())
    def test_shape_and_bounds(self, data):
        N, rules_str, node_names = data
        W = 5

        wrapper = CWSingleStep(rules_str, n_walkers=W)
        states = cp.random.randint(0, 2, size=(N, W)).astype(bool)
        wrapper.set_initial_states(states)

        out = wrapper.simulate_step()
        out_cmp = out.get()

        self.assertEqual(out_cmp.shape, (1, N, W))
        self.assertTrue(np.all((out_cmp == 0) | (out_cmp == 1)))

    @settings(max_examples=10, deadline=None)
    @given(simple_boolean_rules())
    def test_asynchronous_change_limit(self, data):
        N, rules_str, node_names = data
        W = 5

        wrapper = CWSingleStep(rules_str, n_walkers=W)
        states = cp.random.randint(0, 2, size=(N, W)).astype(bool)

        wrapper.set_initial_states(states)

        out = wrapper.simulate_step()
        out_cmp = out.get()
        states_cmp = states.get()

        diffs = (out_cmp[0] != states_cmp)
        self.assertTrue(np.all(diffs.sum(axis=0) <= 1))

    @settings(max_examples=10, deadline=None)
    @given(identity_networks())
    def test_idempotent_identity_rule(self, identity_net):
        N, nodes, rules_str = identity_net
        W = 5

        wrapper = CWSingleStep(rules_str, n_walkers=W)
        states = cp.random.randint(0, 2, size=(N, W)).astype(bool)
        wrapper.set_initial_states(states)

        out = wrapper.simulate_step()
        out_cmp = out.get()
        states_cmp = states.get()

        np.testing.assert_array_equal(out_cmp[0], states_cmp)

# ---------------- Tests for simulate_one_step_aligned_states ----------------
class TestSimulateOneStepAlignedStatesIdentity(unittest.TestCase):
    @settings(max_examples=10, deadline=None)
    @given(identity_networks(), st.integers(min_value=1, max_value=10))
    def test_identity_network_preserves_states(self, identity_net, n_walkers):
        N, nodes, rules = identity_net

        for xp in [np, cp]:
            wrapper = CWSingleStep(rules, n_walkers)
            states = xp.random.randint(0, 2, size=(N, n_walkers)).astype(bool)

            out = simulate_one_step_aligned_states(states, nodes, wrapper)
            out_cmp = out.get() if xp is cp else out
            states_cmp = states.get() if xp is cp else states

            np.testing.assert_array_equal(out_cmp, states_cmp)

if __name__ == "__main__":
    unittest.main()
