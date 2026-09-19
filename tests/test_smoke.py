def test_import():
    import covertext

    assert covertext is not None
    assert covertext.__version__ == "0.1.0"
