"""pytest conftest.py — session-wide fixtures and environment setup.

This file runs before any test collection or import. We use it to block
TensorFlow from being imported by the transformers/sentence_transformers
chain, since TF has a protobuf version conflict in this environment and
TrustLM does not use TF at all.
"""

import os

# Must be set BEFORE sentence_transformers / transformers is imported,
# otherwise transformers will attempt to load TF integration modules
# which fail with a protobuf VersionError.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
