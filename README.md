# CogWorks Week 2 vision benchmark

This repository contains the trusted behavioral benchmark for the CogWorks
facial-recognition and Whispers capstone. It evaluates the application a team
built; it does not rank the shared FaceNet detector or benchmark helper code.

Two benchmark plugins are published:

- `vision-recognition` evaluates known identification and the lifecycle of an
  unknown person who is later enrolled.
- `vision-clustering` evaluates identity clustering with label-invariant
  metrics.

Students may organize their application however they want. A small adapter in
their repository translates that application into the two behavioral
contracts. See [`face_recognition_app`](face_recognition_app/README.md) for the
temporary starter and testing guidance.

## Development

```bash
python -m pip install -e '.[test]'
pytest
ruff check .
python -m build
```

The normal test suite is offline and uses tiny fixtures. CelebA integration is
an explicit workflow because it downloads data and the FaceNet checkpoint.

## Data policy

CelebA images are never committed to this repository. Public manifests pin the
Hugging Face dataset revision and selected rows; official images and labels are
materialized separately in trusted course infrastructure. Use of CelebA is
subject to its non-commercial research agreement.
