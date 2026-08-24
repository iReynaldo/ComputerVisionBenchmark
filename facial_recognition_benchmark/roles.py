"""What Week 2's clustering task asks for, described so a repository can be
searched for it.

The capstone is one task: given a folder of photos, put the ones showing the
same person together. The course names the steps on the way there. Find the
faces in a picture, turn each into a descriptor, measure how close two
descriptors are, build a graph from that, and run whispers over the graph.
Every team writes those, in their own files, under their own names.

So this says what each step does in terms of what goes in and what comes back,
never what it is called. The acceptance test at the bottom is the only thing
that can accept a chain: hand it photos of two people and require that it puts
each person's photos together.

Week 1 taught the shape of this file and one lesson worth repeating: the
validators are for cutting the search, not for judging. A team whose threshold
is badly tuned should be scored badly by the benchmark, not refused by
discovery.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
from typing import Any, List, Sequence

import numpy as np

from cogbench.pipeline import Role, Stage

__all__ = [
    "CLUSTER_ROLE",
    "accepts",
    "looks_like_labels",
    "looks_like_descriptors",
    "write_photos",
]


def looks_like_descriptors(value: Any) -> bool:
    """One vector per face, or a batch of them.

    Deliberately loose about the dimension. The course uses FaceNet's 512, and
    a team who wrote their own encoder is answering the same question with a
    different number.
    """

    if isinstance(value, np.ndarray):
        return value.ndim in (1, 2) and value.size > 0 and value.dtype.kind == "f"
    if isinstance(value, (list, tuple)) and value:
        first = value[0]
        return isinstance(first, np.ndarray) and first.dtype.kind == "f"
    return False


def looks_like_labels(value: Any) -> bool:
    """An answer saying which photos go together.

    Two shapes, both of which the course teaches and the corpus wrote. One
    label per image, in image order, is the obvious one. Groups of nodes is
    the other: the capstone's `connected_components` "returns the groups"
    (week2-vision-capstone.md:392), and two of the three audited teams end
    there rather than flattening back to a list.

    A label is whatever a team used to mean "these two are the same person": a
    number, a name, a node id. What matters is that the answer can be turned
    into a grouping, which is all the metric reads.
    """

    if isinstance(value, np.ndarray):
        return value.ndim == 1 and value.size > 0
    if not isinstance(value, (list, tuple)) or not value:
        return False
    if all(isinstance(item, (int, str, np.integer)) for item in value):
        return True
    # Groups: a list of lists, each holding whatever their node class is.
    return all(isinstance(item, (list, tuple, set)) and len(item) > 0 for item in value)


#: Images in, one label per image out.
#:
#: Every step between is marked fusible, because teams divide this work
#: differently and all the divisions are correct. One 2026 team wrote
#: `whispers(images, iterations)` doing the whole thing; another wrote
#: `get_descriptor`, `adj_list`, and `whispers` separately; a third put
#: detection and description in one `process_image`. Insisting on any one
#: division would refuse working code over how it was organized.
def looks_like_graph(value: Any) -> bool:
    """Nodes, or nodes and their adjacency.

    The course's own design is a graph: "a list of nodes and an adjacency
    graph that describes the weighted connections between your nodes"
    (docs/capstones/week2-vision-capstone.md:386). Two of three audited teams
    return exactly that, so it is a step, not an implementation detail.

    Deliberately structural rather than typed: a node is whatever class they
    wrote, and requiring a shape would be requiring their design.
    """

    if isinstance(value, tuple) and len(value) == 2:
        return looks_like_graph(value[0])
    if isinstance(value, dict):
        return bool(value)
    if not isinstance(value, (list, tuple)) or not value:
        return False
    first = value[0]
    # Not a descriptor and not a label: something they built.
    return not isinstance(first, (int, str, float, np.integer, np.ndarray))


CLUSTER_ROLE = Role(
    "cluster",
    (
        Stage(
            "descriptors",
            prefers=("descriptor", "embed", "encode", "facenet", "process"),
            produces=looks_like_descriptors,
            arity=1,
            tunings=(0.3, 0.4, 0.5, 0.6, 0.7),
            # A team whose graph builder reads the photos itself never had a
            # separate descriptor step. One audited repository's
            # `adj_list(image_paths, threshold)` describes every face and
            # builds the graph in one function, which is the same work in one
            # place rather than two.
            fusible=True,
            # The capstone hands one photo at a time
            # (docs/capstones/week2-vision-capstone.md:176), and every audited
            # team wrote a per-photo descriptor function.
            per_item=True,
        ),
        Stage(
            "graph",
            prefers=("adj", "node", "graph", "build", "connect"),
            produces=looks_like_graph,
            # A team who went straight from descriptors to labels never built
            # one, and that is a complete answer too.
            fusible=True,
            tunings=(0.3, 0.4, 0.5, 0.6, 0.7),
        ),
        Stage(
            "labels",
            prefers=("whisper", "cluster", "label", "component", "group"),
            produces=looks_like_labels,
            fusible=True,
            # Whispers is iterative and the course does not fix a count, so
            # their `whispers(nodes, adj, iterations)` takes one with no
            # default. The cutoffs are here too, for a team who clusters
            # straight from descriptors and takes the threshold at this step.
            #
            # Enough passes to settle on this fixture. The benchmark's own
            # driver decides how long a scored run gets; this only has to be
            # long enough to tell a working chain from a broken one.
            tunings=(10, 20, 0.3, 0.4, 0.5, 0.6, 0.7),
            # The course says `propagate_label` "should update that node's
            # label" and has `whispers` record how the component count
            # changes as it runs (week2-vision-capstone.md:390-392). One
            # audited team's whispers does exactly that: it returns the
            # counts, and the labels are on the graph it was handed.
            in_place=True,
        ),
    ),
)


def accepts(chain, images, expected):
    """Cluster photos of two people and require the two groups back.

    The weakest test that still kills a wrong binding. It does not measure
    quality: a team whose threshold splits one person into two clusters should
    lose points on the benchmark, not be refused here. What it rejects is a
    chain that returns the same label for everyone, or a different label for
    every photo, both of which a wrongly assembled pipeline produces.

    That is what it now does. It used to require the exact partition, which
    read as the same rule and is a much stronger one: it refused a pipeline
    that grouped the photos nearly right, and refused it with a wiring
    message. A grouping that is neither degenerate is an answer to this
    question, and how good an answer belongs to the metric.

    Returns ``(passed, detail)``.
    """

    with _fresh_state():
        try:
            labels = _run(chain, images)
        except BaseException as error:  # noqa: BLE001 - student code raises anything
            return False, "clustering raised {}: {}".format(
                type(error).__name__, str(error)[:120]
            )

    got = _as_grouping(labels, len(images))
    if got is None:
        return False, "returned {} that does not say which photos go together".format(
            type(labels).__name__
        )
    want = _grouping(expected)
    if got == want:
        return True, "grouped {} photos into {} people".format(len(images), len(want))

    # Not an exact match, which is not the same as not an answer. The
    # docstring above promised to reject a chain that says everyone is the
    # same person or that everyone is different, and to leave quality to the
    # benchmark. Requiring the exact partition broke that promise: it also
    # refused a chain that grouped the photos nearly right, which is a tuning
    # result and the single most useful thing a team can be told.
    #
    # Measured on one 2026 repository. Its pipeline binds and runs, and at
    # every cutoff the search tries it returns 4, 5 or 6 groups where the
    # fixture has 3, because its threshold splits one person in two. Refusing
    # that reported "nothing took that for the labels step", which is false
    # and sends them to look for a function they already wrote.
    #
    # So the bar here is: did this chain answer the question at all. Two
    # degenerate answers are still refused, because both are what a wrongly
    # assembled pipeline produces rather than what a working one gets wrong.
    if len(got) <= 1:
        return False, "put all {} photos in one group".format(len(images))
    if len(got) >= len(images):
        return False, "put every photo in its own group"
    return True, "grouped {} photos into {} groups where there are {} people".format(
        len(images), len(got), len(want)
    )


def _as_grouping(answer: Any, count: int):
    """Read an answer into "which photos went together", or None.

    Two shapes. One label per photo, in photo order, is read positionally.
    Groups of nodes are read by asking each node which photo it is, which the
    course's own node class answers: it carries the file path of its image
    (week2-vision-capstone.md:418). Anything else is not an answer to this
    question and is refused rather than guessed at.
    """

    if isinstance(answer, np.ndarray):
        answer = answer.tolist()
    if not isinstance(answer, (list, tuple)) or not answer:
        return None

    if all(isinstance(item, (int, str, np.integer)) for item in answer):
        return _grouping(answer) if len(answer) == count else None

    groups = []
    for group in answer:
        if not isinstance(group, (list, tuple, set)):
            return None
        members = frozenset(_identifies(node) for node in group)
        if any(member is None for member in members):
            return None
        groups.append(members)
    if sum(len(group) for group in groups) != count:
        return None
    return frozenset(groups)


def _identifies(node: Any):
    """Which photo a node stands for, read off the node itself.

    The course's node carries "the file path of the image corresponding to
    this node" (week2-vision-capstone.md:438), so that is what is asked for.
    A node that says nothing about its photo cannot be placed, and the answer
    is refused rather than guessed at.
    """

    for attribute in ("image_path", "file_path", "path", "filepath", "image", "id", "ID"):
        value = getattr(node, attribute, None)
        if value is None:
            continue
        return str(value)
    return None


def _grouping(labels: Sequence[Any]) -> frozenset:
    """Which photos went together, independent of what the groups are called."""

    groups: dict = {}
    for index, label in enumerate(labels):
        groups.setdefault(label, []).append(index)
    return frozenset(frozenset(members) for members in groups.values())


def _run(chain: Sequence[Any], images: Sequence[Any]) -> Any:
    """Push images through a resolved chain, changing nothing on the way."""

    value = chain[0].call(images)
    for step in chain[1:]:
        value = step.call(value)
    return value


def write_photos(images: Sequence[Any]) -> List[Any]:
    """The same photos, written out as PNG files, returning their paths.

    The capstone document tells students to write "a function that takes in a
    list of image-paths" (docs/capstones/week2-vision-capstone.md:386), and
    all three audited 2026 repositories did. The benchmark hands over arrays,
    so a team that followed the instructions had no function the search could
    call, and every one of them was refused at the first step.

    Which is our contract's problem, not their code's. The photos are the same
    photos either way; a file is one of the two shapes the course taught, so
    the search offers both and their own function decides which it takes.
    """

    from PIL import Image

    # Kept for the life of the process rather than deleted after the probe:
    # the search calls their function many times, and a path that stopped
    # existing between calls would look like their code failing.
    from pathlib import Path as _Path

    folder = tempfile.mkdtemp(prefix="cogworks-week2-photos-")
    paths = []
    for index, image in enumerate(images):
        path = _Path(folder) / "photo_{:03d}.png".format(index)
        Image.fromarray(np.asarray(image, dtype=np.uint8)).save(path)
        # Path objects, not strings. One audited repository does
        # `p.parent` on what it is given, and a string has no `.parent`, so a
        # str fixture failed their code over how we spelled a path rather than
        # over anything they wrote. `str(path)` works on a Path everywhere the
        # corpus uses one; the reverse is not true.
        paths.append(path)
    return paths


@contextlib.contextmanager
def _fresh_state():
    """One attempt, one empty directory, and no narration.

    Their functions write: one audited repository sorts photos into a `result`
    directory as it clusters. And they print, at length. Neither belongs in the
    search for which of their functions to call.
    """

    previous = os.getcwd()
    saved_out, saved_err = sys.stdout, sys.stderr
    with tempfile.TemporaryDirectory(prefix="cogworks-week2-") as temporary:
        os.chdir(temporary)
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        try:
            yield
        finally:
            sys.stdout, sys.stderr = saved_out, saved_err
            os.chdir(previous)
