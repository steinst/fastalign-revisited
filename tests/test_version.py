import fastalign


def test_version():
    assert isinstance(fastalign.__version__, str) and fastalign.__version__
