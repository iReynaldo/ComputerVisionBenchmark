# Architecture and trust boundary

The capstone asks students to build a facial-recognition application and a
Whispers clustering implementation. The benchmark therefore stays outside the
student application's architecture and observes only the behaviors described
by the [facial-recognition capstone][recognition] and [Whispers assignment][whispers].

Two `cogworks.submissions.v2` factories receive the course-owned FaceNet model.
The recognition adapter enrolls opaque person labels and returns a label or
`None` for each one-face RGB array. The clustering adapter returns an arbitrary
string or integer cluster label for each array. Cluster names have no meaning.

This boundary deliberately does not require `Profile`, `Database`, `Node`,
pickle files, a result dictionary, or a specific module layout. Reynaldo's
original broad `FaceRecognitionApp` methods remain a bounded compatibility
surface, but the benchmark never guesses how an unfamiliar method should map.
Its error reports the methods it found and points to `benchmark_adapter.py`.

The benchmark creates a fresh adapter for each scenario. A recognition scenario
retains that one adapter across initial enrollment, known queries, unknown
rejection, enrollment of the new identity, and held-out re-identification. This
makes database updates observable without prescribing how they are implemented.

## Publication boundary

The public repository contains contracts, drivers, deterministic scorers,
public manifests, and an interface-only temporary starter. It contains no
distance rule, database implementation, or Whispers solution. The completed
application used for calibration remains instructor-only. The temporary starter
will later be extracted into `CogWorksBWSI/week2-vision-capstone` with clean
history and registered by immutable repository ID and commit.

This records the later course decisions from both class transcripts: distribute
an interface rather than the completed application, keep a separate forkable
student template, verify the integrated path jointly, and cover macOS and
Windows in addition to Linux. It also preserves the broader two-track decision
without forcing every team into one application class.

[recognition]: https://rsokl.github.io/CogWeb/Video/FacialRecognition.html
[whispers]: https://rsokl.github.io/CogWeb/Video/Whispers.html
