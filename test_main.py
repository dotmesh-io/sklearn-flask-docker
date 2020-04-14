"""Tests for main.py."""

import os
import tempfile
import tarfile
import shutil

import pytest

import main

ORIGINAL_DIR = os.getcwd()


@pytest.fixture
def client():
    """Create a client, and run out of temporary directory."""
    main.app.config["TESTING"] = True
    tempdir = tempfile.mkdtemp()
    os.environ["MODEL_JOBLIB_FILE"] = os.path.join(tempdir, "mymodel")
    os.chdir(tempdir)
    try:
        with main.app.test_client() as client:
            yield client
    finally:
        os.chdir(ORIGINAL_DIR)


def test_default_predict(client):
    """The default prediction logic, loading a pickled sklearn model."""
    shutil.copy(os.path.join(ORIGINAL_DIR, "example_model", "model.joblib"), "mymodel")

    main.setup()
    result = client.post("/v1/models/model:predict", json={"instances": [[1, 2, 3, 4]]})
    assert result.json["predictions"] == [
        [pytest.approx(0.26, 0.1), pytest.approx(0.23, 0.1), pytest.approx(0.51, 0.1)]
    ]


def test_default_predict_from_tarball(client):
    """
    The default prediction logic, loading a pickled sklearn model from a
    tarball.
    """
    tf = tarfile.open("mymodel", "w")
    tf.add(
        os.path.join(ORIGINAL_DIR, "example_model", "model.joblib"),
        arcname="mymodel/mymodel",
    )
    tf.close()

    main.setup()
    result = client.post("/v1/models/model:predict", json={"instances": [[1, 2, 3, 4]]})
    assert result.json["predictions"] == [
        [pytest.approx(0.26, 0.1), pytest.approx(0.23, 0.1), pytest.approx(0.51, 0.1)]
    ]
