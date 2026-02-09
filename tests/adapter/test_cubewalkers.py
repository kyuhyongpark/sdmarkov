import unittest
import numpy as np
import cupy as cp
from sdmarkov.adapter.cubewalkers import CWSingleStep, simulate_one_step_aligned_states


class TestCWSingleStep(unittest.TestCase):
    def setUp(self):
        # Tiny deterministic network: 3 nodes
        self.rules = """
        X, !X
        Y, X & !Z
        Z, Y
        """
        self.n_walkers = 5

    def test_set_initial_states(self):
        wrapper = CWSingleStep(self.rules, self.n_walkers)
        states = cp.zeros((3, self.n_walkers)).astype(bool)
        wrapper.set_initial_states(states)
        model_states = wrapper.model.initial_states
        cp.testing.assert_array_equal(model_states, states)

    def test_set_initial_states_wrong_shape(self):
        wrapper = CWSingleStep(self.rules, self.n_walkers)
        states = cp.zeros((2, self.n_walkers)).astype(bool)
        with self.assertRaises(ValueError):
            wrapper.set_initial_states(states)

    def test_simulate_step_shape(self):
        wrapper = CWSingleStep(self.rules, self.n_walkers)
        states = cp.zeros((3, self.n_walkers)).astype(bool)
        wrapper.set_initial_states(states)
        traj = wrapper.simulate_step()
        self.assertEqual(traj.shape, (1, 3, self.n_walkers))


class TestSingleStepAlignedStates(unittest.TestCase):
    def setUp(self):
        self.rules = """
        X, X
        Y, Y
        Z, Z
        """
        self.n_walkers = 4
        self.wrapper = CWSingleStep(self.rules, self.n_walkers)
        self.sampler_nodes = ["Z", "X", "Y"]  # deliberately permuted
        self.states = np.zeros((3, self.n_walkers), dtype=bool)
        # set a simple pattern for clarity
        self.states[0, :] = 1  # Z row all ones
        self.states[1, 0] = 1  # X row: first walker 1
        self.states[2, 1] = 1  # Y row: second walker 1

    def test_output_shape(self):
        out = simulate_one_step_aligned_states(
            self.states, self.sampler_nodes, self.wrapper,
        )
        self.assertEqual(out.shape, (3, self.n_walkers))

    def test_node_order_preserved(self):
        out = simulate_one_step_aligned_states(
            self.states, self.sampler_nodes, self.wrapper,
        )

        # The row sum should match the original rows because node order preserved
        np.testing.assert_array_equal(out.sum(axis=1), self.states.sum(axis=1))


    def test_backend_cp(self):
        states_cp = cp.array(self.states)
        out = simulate_one_step_aligned_states(
            states_cp, self.sampler_nodes, self.wrapper,
        )
        self.assertEqual(out.shape, (3, self.n_walkers))
        # convert to numpy for easy checking
        out_np = cp.asnumpy(out)
        self.assertTrue(np.all(out_np.sum(axis=1) >= 0))  # trivial sanity


if __name__ == "__main__":
    unittest.main()
