"""Discovery must score a repository the way a human reading it did.

One Week 2 repository in the 2026 corpus has an instructor-written adapter.
Its clustering score is the only ground truth for whether the automatic
binding found the right functions and ran them the right way. The adapter
pads a photo with no detected face to a zero descriptor so it stays in the
graph; the discovered chain runs their `adj_list`, which drops it. On the
public test tier every face is detected, so the two agree exactly there.

Skipped when the corpus is not checked out.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CORPUS = ROOT / ".cache" / "student-repos"
ADAPTERS = ROOT / "benchmarks" / "adapters"

sys.path.insert(0, str(ROOT / "python" / "cogbench" / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "week2"))


@unittest.skipUnless(CORPUS.is_dir(), "student corpus is not checked out")
class DiscoveryMatchesTheHandWrittenAdapter(unittest.TestCase):
    def test_lashika_vision_module_capstone(self) -> None:
        from cogbench.plugins import load_benchmark
        from cogbench.resolve import resolve
        from cogbench.runner import _facenet_model

        name = "LashikaKapoor28__Vision_Module_Capstone"
        repo = (CORPUS / name).resolve()
        adapter = ADAPTERS / name / "submission.py"
        if not repo.is_dir() or not adapter.is_file():
            self.skipTest("{} is not in this checkout".format(name))

        plugin = load_benchmark("vision-clustering")
        spec = plugin.discovery()
        found = resolve(
            repo,
            chain_role=spec.chain_role,
            fixture=spec.fixture,
            accepts=spec.accepts,
            arrangements=spec.arrangements,
            hints=spec.hints,
            benchmark="vision-clustering",
        )
        self.assertTrue(found.ready, found.verdict.headline)
        self.assertEqual(
            [step.label for step in found.chain],
            ["whispers.adj_list", "whispers.whispers", "whispers.connected_comps"],
        )

        cases = plugin.load_cases("test")
        model = _facenet_model()
        discovered = plugin.score(
            plugin.run(lambda *a, **k: plugin.submission_from_discovery(found), model, cases),
            cases,
        )

        sys.path.insert(0, str(repo))
        loaded = importlib.util.spec_from_file_location("lashika_adapter", adapter)
        module = importlib.util.module_from_spec(loaded)
        loaded.loader.exec_module(module)
        human = plugin.score(plugin.run(module.create_clustering_adapter, model, cases), cases)

        self.assertAlmostEqual(
            discovered["clustering_pairwise_f1"], human["clustering_pairwise_f1"], places=4
        )


if __name__ == "__main__":
    unittest.main()
