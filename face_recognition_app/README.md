# Temporary Week 2 vision starter

This is an interface-only starter for the CogWorks vision capstone. It contains
no recognition, database, distance, or Whispers implementation. The subtree
will move to a separate forkable course repository after the benchmark is
reviewed.

You own the design of your application. Keep the optional broad
`FaceRecognitionApp`, replace it with functions, introduce your own classes, or
build something else. Only the two small factories in `benchmark_adapter.py`
are fixed integration points.

## Set up

```bash
python -m pip install -e '.[test]'
pytest
```

The starter tests pass before the capstone algorithms are implemented because
they check wiring only. Add tests for your own architecture and features as you
work. Those tests remain in your repository and never affect leaderboard
scores.

## Benchmark entry points

- `vision-recognition` must return an object with `enroll` and `recognize`.
- `vision-clustering` must return an object with `cluster`.
- Each factory receives the benchmark-owned FaceNet model.

Both factories may return wrappers around one shared application design or two
separate adapters. Read [TESTING.md](TESTING.md) before running real data.
