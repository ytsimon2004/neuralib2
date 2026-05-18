from neuralib.atlas.cellatlas import load_cellatlas


def test_load():
    df = load_cellatlas()
    print(df)
