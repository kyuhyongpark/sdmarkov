import unittest
from hypothesis import given, settings
from hypothesis import strategies as st

# Import the functions under test
# adjust imports to match your package structure
from sdmarkov.succession_diagram import (
    trap_spaces_overlap,
    intersect_trap_spaces,
    close_trap_spaces_under_intersection,
    assign_states_to_trap_spaces
)

# ---------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------

NODE_NAMES = ["A", "B", "C", "D"]

trap_space_strategy = st.dictionaries(
    keys=st.sampled_from(NODE_NAMES),
    values=st.sampled_from([0, 1]),
    max_size=len(NODE_NAMES),
)

trap_space_list_strategy = st.lists(
    trap_space_strategy,
    min_size=1,
    max_size=6,
)


# ---------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------

class TestTrapSpaceProperties(unittest.TestCase):
    """
    Property-based tests for symbolic trap-space logic.

    These tests verify invariants that must hold for *all*
    valid trap-space inputs.
    """

    # -----------------------------
    # Overlap properties
    # -----------------------------

    @given(trap_space_strategy)
    def test_overlap_reflexive(self, ts):
        """A trap space always overlaps with itself."""
        self.assertTrue(trap_spaces_overlap(ts, ts))

    @given(trap_space_strategy, trap_space_strategy)
    def test_overlap_symmetric(self, a, b):
        """Overlap is symmetric."""
        self.assertEqual(
            trap_spaces_overlap(a, b),
            trap_spaces_overlap(b, a),
        )

    # -----------------------------
    # Intersection properties
    # -----------------------------

    @given(trap_space_strategy, trap_space_strategy)
    def test_intersection_consistency(self, a, b):
        """
        If intersection exists, it must agree with both inputs.
        """
        inter = intersect_trap_spaces(a, b)

        if inter is None:
            # inconsistent intersections must not overlap
            self.assertFalse(trap_spaces_overlap(a, b))
        else:
            # intersection must overlap with both
            self.assertTrue(trap_spaces_overlap(a, inter))
            self.assertTrue(trap_spaces_overlap(b, inter))

            # intersection must be a superset of assignments
            for k, v in inter.items():
                if k in a:
                    self.assertEqual(a[k], v)
                if k in b:
                    self.assertEqual(b[k], v)

    # -----------------------------
    # Closure properties
    # -----------------------------

    @given(trap_space_list_strategy)
    @settings(max_examples=200)
    def test_closure_is_idempotent(self, trap_spaces):
        """
        Closing twice does not change the result.
        """
        closed_once = close_trap_spaces_under_intersection(trap_spaces)
        closed_twice = close_trap_spaces_under_intersection(closed_once)

        self.assertEqual(
            set(map(frozenset, closed_once)),
            set(map(frozenset, closed_twice)),
        )

    @given(trap_space_list_strategy)
    def test_closure_contains_originals(self, trap_spaces):
        """
        Closure must not remove existing trap spaces.
        """
        closed = close_trap_spaces_under_intersection(trap_spaces)

        closed_set = {frozenset(ts.items()) for ts in closed}
        for ts in trap_spaces:
            self.assertIn(frozenset(ts.items()), closed_set)


    @given(trap_space_list_strategy)
    def test_closure_is_closed(self, trap_spaces):
        """
        After closure, all overlapping pairs have their intersection included.
        """
        closed = close_trap_spaces_under_intersection(trap_spaces)
        closed_set = {frozenset(ts.items()) for ts in closed}

        for i, a in enumerate(closed):
            for b in closed[i + 1:]:
                if trap_spaces_overlap(a, b):
                    inter = intersect_trap_spaces(a, b)
                    self.assertIsNotNone(inter)
                    self.assertIn(frozenset(inter.items()), closed_set)



# ---------- Strategies ----------

@st.composite
def nodes_and_closed_trap_spaces(draw):
    n = draw(st.integers(min_value=1, max_value=5))
    nodes = [f"X{i}" for i in range(n)]

    def trap_space_strategy():
        return st.dictionaries(
            keys=st.sampled_from(nodes),
            values=st.integers(min_value=0, max_value=1),
            max_size=n,
        )

    raw = draw(
        st.lists(
            trap_space_strategy(),
            min_size=1,
            max_size=6,
            unique_by=lambda d: frozenset(d.items()),
        )
    )

    # Ensure {} is present
    if {} not in raw:
        raw.append({})

    closed = close_trap_spaces_under_intersection(raw)

    # Enforce {} survives closure
    assert {} in closed

    return nodes, closed


# ---------- Helpers ----------

def state_satisfies_trap_space(state, nodes, ts):
    for i, node in enumerate(nodes):
        if node in ts and int(state[i]) != ts[node]:
            return False
    return True


def is_more_specific(a, b):
    """
    a strictly more specific than b
    """
    return len(a) > len(b) and all(
        k in a and a[k] == v for k, v in b.items()
    )


# ---------- Property tests ----------

class TestAssignStatesProperties(unittest.TestCase):

    @given(nodes_and_closed_trap_spaces())
    def test_full_coverage(self, data):
        nodes, trap_spaces = data
        groups = assign_states_to_trap_spaces(nodes, trap_spaces)

        total = sum(len(g) for g in groups)
        self.assertEqual(total, 2 ** len(nodes))

    @given(nodes_and_closed_trap_spaces())
    def test_unique_assignment(self, data):
        nodes, trap_spaces = data
        groups = assign_states_to_trap_spaces(nodes, trap_spaces)

        all_states = [s for g in groups for s in g]
        self.assertEqual(len(all_states), len(set(all_states)))

    @given(nodes_and_closed_trap_spaces())
    def test_soundness(self, data):
        nodes, trap_spaces = data
        groups = assign_states_to_trap_spaces(nodes, trap_spaces)

        for ts, states in zip(trap_spaces, groups):
            for s in states:
                self.assertTrue(
                    state_satisfies_trap_space(s, nodes, ts)
                )

    @given(nodes_and_closed_trap_spaces())
    def test_priority_correctness(self, data):
        """
        No state is assigned to a trap space if a strictly
        more specific satisfied trap space exists.
        """
        nodes, trap_spaces = data
        groups = assign_states_to_trap_spaces(nodes, trap_spaces)

        for i, (ts, states) in enumerate(zip(trap_spaces, groups)):
            for s in states:
                for other in trap_spaces:
                    if is_more_specific(other, ts):
                        if state_satisfies_trap_space(s, nodes, other):
                            self.fail(
                                f"State {s} assigned to {ts}, "
                                f"but more specific {other} also matches"
                            )


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
