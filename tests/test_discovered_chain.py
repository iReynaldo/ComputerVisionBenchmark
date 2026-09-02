"""A discovered chain is run the way the search proved it works.

The search binds `adj_list(paths, threshold)` by offering a cutoff, and binds
`whispers(nodes, adj, iterations)` by spreading the pair `adj_list` returned.
The acceptance test and the scored run must make those same calls, or they
score a different program than the one that passed. Measured on one 2026
repository before this existed: the acceptance test called `adj_list(paths)`,
got a TypeError, and reported that their code ran and answered wrongly.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python" / "cogbench" / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "week2"))

from cogbench.pipeline import Candidate  # noqa: E402

from facial_recognition_benchmark.roles import _run  # noqa: E402


def _module(name, source):
    module = ModuleType(name)
    exec(compile(source, name, "exec"), module.__dict__)
    return module


THEIRS = _module(
    "theirs",
    "class Node:\n"
    "    def __init__(self, image_path, label):\n"
    "        self.image_path = image_path\n"
    "        self.label = label\n"
    "def adj_list(paths, threshold):\n"
    "    nodes = [Node(p, i) for i, p in enumerate(paths)]\n"
    "    return nodes, {n: [] for n in nodes}\n"
    "def whispers(nodes, adj, iterations):\n"
    "    return [len(adj)] * iterations\n"
    "def connected_comps(adj, nodes):\n"
    "    return [[n] for n in nodes]\n",
)


class DiscoveredChainTests(unittest.TestCase):
    def test_a_step_bound_with_a_tuning_is_called_with_it(self):
        chain = [Candidate("theirs.adj_list", THEIRS.adj_list, "theirs", tuning=0.3)]

        nodes, adj = _run(chain, ["a.png", "b.png"])

        self.assertEqual([n.image_path for n in nodes], ["a.png", "b.png"])
        self.assertEqual(len(adj), 2)

    def test_a_step_bound_without_a_tuning_raises_as_it_did_before(self):
        chain = [Candidate("theirs.adj_list", THEIRS.adj_list, "theirs")]

        with self.assertRaises(TypeError):
            _run(chain, ["a.png"])

    def test_a_returned_pair_is_spread_into_the_next_step(self):
        chain = [
            Candidate("theirs.adj_list", THEIRS.adj_list, "theirs", tuning=0.3),
            Candidate("theirs.whispers", THEIRS.whispers, "theirs", tuning=10),
        ]

        self.assertEqual(_run(chain, ["a.png", "b.png"]), [2] * 10)

    def test_a_returned_pair_is_spread_reversed_when_that_is_their_order(self):
        chain = [
            Candidate("theirs.adj_list", THEIRS.adj_list, "theirs", tuning=0.3),
            Candidate("theirs.connected_comps", THEIRS.connected_comps, "theirs"),
        ]

        groups = _run(chain, ["a.png", "b.png"])

        self.assertEqual([[n.image_path for n in g] for g in groups], [["a.png"], ["b.png"]])


if __name__ == "__main__":
    unittest.main()


class TheScoredRunPresentsTheBoundForm(unittest.TestCase):
    """A chain bound on paths is scored on paths; one bound on arrays, on
    arrays. The search proved one of the two, and scoring the other scores a
    function that never ran."""

    def _images(self):
        import numpy as np

        return [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(3)]

    def test_a_path_taking_first_step_is_given_paths(self):
        from facial_recognition_benchmark.discovered import DiscoveredClustering

        seen = []

        def takes_paths(paths):
            seen.append([type(p).__name__ for p in paths])
            return [str(p) for p in paths]

        takes_paths.__module__ = "theirs"
        chain = [Candidate("theirs.takes_paths", takes_paths, "theirs", form=1)]

        labels = DiscoveredClustering(chain).cluster(self._images())

        self.assertEqual(len(labels), 3)
        self.assertEqual(seen, [["PosixPath"] * 3])

    def test_an_array_taking_first_step_is_given_arrays(self):
        from facial_recognition_benchmark.discovered import DiscoveredClustering

        seen = []

        def takes_arrays(images):
            seen.append([type(i).__name__ for i in images])
            return list(range(len(images)))

        takes_arrays.__module__ = "theirs"
        chain = [Candidate("theirs.takes_arrays", takes_arrays, "theirs", form=0)]

        DiscoveredClustering(chain).cluster(self._images())

        self.assertEqual(seen, [["ndarray"] * 3])


class GroupsBecomeLabelsInPhotoOrder(unittest.TestCase):
    """The driver wants one label per photo; two audited teams return groups
    of nodes. Reading them back is placement, not repair."""

    def test_groups_are_placed_by_the_photo_each_node_names(self):
        from facial_recognition_benchmark.roles import labels_in_photo_order

        class N:
            def __init__(self, p):
                self.image_path = p

        photos = ["a.png", "b.png", "c.png", "d.png"]
        answer = [[N("c.png"), N("a.png")], [N("d.png"), N("b.png")]]

        labels = labels_in_photo_order(answer, photos)

        self.assertEqual(len(labels), 4)
        self.assertEqual(labels[0], labels[2])
        self.assertEqual(labels[1], labels[3])
        self.assertNotEqual(labels[0], labels[1])

    def test_a_dropped_photo_gets_a_label_of_its_own(self):
        from facial_recognition_benchmark.roles import labels_in_photo_order

        class N:
            def __init__(self, p):
                self.image_path = p

        labels = labels_in_photo_order([[N("a.png"), N("b.png")]], ["a.png", "b.png", "c.png"])

        self.assertEqual(labels[0], labels[1])
        self.assertNotEqual(labels[2], labels[0])

    def test_a_positional_list_passes_through(self):
        from facial_recognition_benchmark.roles import labels_in_photo_order

        self.assertEqual(labels_in_photo_order([0, 0, 1], ["a", "b", "c"]), [0, 0, 1])

    def test_groups_that_name_no_photo_are_refused(self):
        from facial_recognition_benchmark.roles import labels_in_photo_order

        with self.assertRaises(TypeError):
            labels_in_photo_order([[object()], [object()]], ["a", "b"])

    def test_the_scored_run_returns_one_label_per_photo(self):
        from facial_recognition_benchmark.discovered import DiscoveredClustering

        def groups(paths):
            return [[THEIRS.Node(p, 0) for p in paths[:2]], [THEIRS.Node(p, 1) for p in paths[2:]]]

        groups.__module__ = "theirs"
        chain = [Candidate("theirs.groups", groups, "theirs", form=1)]
        import numpy as np

        labels = DiscoveredClustering(chain).cluster([np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(4)])

        self.assertEqual(len(labels), 4)
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[2], labels[3])
        self.assertNotEqual(labels[0], labels[2])


class ACourseStyleNodeNamesItsPhotoByIndex(unittest.TestCase):
    def test_integer_ids_are_read_as_positions(self):
        from facial_recognition_benchmark.roles import labels_in_photo_order

        class N:
            def __init__(self, i):
                self.id = i
                self.file_path = None

        labels = labels_in_photo_order([[N(0), N(2)], [N(1)]], ["a.png", "b.png", "c.png"])

        self.assertEqual(labels[0], labels[2])
        self.assertNotEqual(labels[0], labels[1])


class TheScoredRunIsSeeded(unittest.TestCase):
    def test_the_same_seed_gives_the_same_labels(self):
        import random

        from facial_recognition_benchmark.discovered import DiscoveredClustering

        def draws(paths):
            return [random.randrange(1000) for _ in paths]

        draws.__module__ = "theirs"
        chain = [Candidate("theirs.draws", draws, "theirs", form=0)]
        images = [b"x", b"y", b"z"]

        first = DiscoveredClustering(chain).cluster(images, seed=7)
        second = DiscoveredClustering(chain).cluster(images, seed=7)
        other = DiscoveredClustering(chain).cluster(images, seed=8)

        self.assertEqual(first, second)
        self.assertNotEqual(first, other)
