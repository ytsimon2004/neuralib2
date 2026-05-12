import datetime

from neuralib.persistence.persistence import field, load, persistence_class, save
from neuralib.persistence.validator import _to_datetime, create_date_validate


@persistence_class
class Cache:
    animal: str = field(filename=True, validator=True)
    value: int


def test_save_and_load_accept_str_path(tmp_path):
    cache = Cache('A00')
    cache.value = 42
    output = tmp_path / 'cache.pkl'

    save(cache, str(output))
    loaded = load(Cache, str(output))

    assert loaded.animal == 'A00'
    assert loaded.value == 42


def test_to_datetime_preserves_datetime_and_promotes_date():
    date = datetime.date(2026, 5, 12)
    dt = datetime.datetime(2026, 5, 12, 13, 30)

    assert _to_datetime(dt) is dt
    assert _to_datetime(date) == datetime.datetime(2026, 5, 12)


def test_create_date_validate_uses_reference_timestamp():
    assert create_date_validate(
        datetime.datetime(2026, 5, 12),
        datetime.datetime(2026, 5, 11),
        verbose=False,
    )
