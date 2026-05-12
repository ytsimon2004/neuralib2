from bokeh.document import Document

from neuralib.dashboard.tool import TimeoutUpdateValue


def test_timeout_update_value_accepts_none_as_value():
    values = []
    updater = TimeoutUpdateValue[None](Document(), values.append)

    updater.update(None)
    updater.callback()

    assert values == [None]


def test_timeout_update_value_callback_consumes_pending_value():
    values = []
    updater = TimeoutUpdateValue[int](Document(), values.append)

    updater.update(1)
    updater.callback()
    updater.callback()

    assert values == [1]
