"""Prove all independent CountFilter state using each actual alternate RouteB output."""

from copy import deepcopy

import numpy as np


def verify_count_state(template, selector):
    stock, alternate = selector.model.cf, selector.outer.cf
    native = deepcopy(template.cf)
    native.reset_state()
    n = selector.summary()["alternate_rows"]
    for row, (event, p) in enumerate(zip(selector.events[:n], selector.outer.raw, strict=True)):
        result = native.step(event[0], p)
        np.testing.assert_array_equal(result, selector.outer.count[row])
    if native.t != alternate.t or native.latch != alternate.latch or native.dead or alternate.dead:
        raise ValueError("native independent count state or latch differs")
    for key in ("V", "al", "cum", "blk"):
        expected, current = getattr(native, key), getattr(alternate, key)
        np.testing.assert_array_equal(expected, current)
        if key != "V" and np.shares_memory(getattr(stock, key), current):
            raise ValueError("native alternate urn or likelihood aliases the stock state")
    for a, b in zip(stock.V[:n], alternate.V, strict=True):
        np.testing.assert_array_equal(a, b)
        if np.shares_memory(a, b):
            raise ValueError("alternate view history aliases stock")
    return dict(exact_alternate_count_state=True, separate_count_urn=True,
                stock_count_latch=stock.latch, alternate_count_latch=alternate.latch, count_rows=n)
