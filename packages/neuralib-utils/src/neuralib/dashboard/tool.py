from typing import Generic, TypeVar, Callable

from bokeh.document import Document

__all__ = ['TimeoutUpdateValue']

T = TypeVar('T')


class _Missing:
    pass


missing = _Missing()


class TimeoutUpdateValue(Generic[T]):
    def __init__(self,
                 document: Document,
                 callback: Callable[[T], None],
                 delay=1000):
        self.__document = document
        self.__value: T | _Missing = missing
        self.__callback = callback
        self.delay = delay

    def update(self, value: T):
        old_value = self.__value
        self.__value = value
        if old_value is missing:
            self.__document.add_timeout_callback(self.callback, self.delay)

    def callback(self):
        value = self.__value
        self.__value = missing
        if not isinstance(value, _Missing):
            self.__callback(value)
