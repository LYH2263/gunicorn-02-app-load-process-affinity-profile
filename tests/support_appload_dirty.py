#
# This file is part of gunicorn released under the MIT license.
# See the NOTICE for more information.

"""Exclusive dirty app support for the application-load profile tests.

This module is deliberately separate from ``support_dirty_app.py`` /
``support_dirty_apps.py``: nothing else in the test suite imports it.
That lets the out-of-process loading tests assert, from an HTTP-side
process, that the dirty app module was never imported there even though
it is reachable through ``DirtyClient``.
"""

import os

from gunicorn.dirty.app import DirtyApp


class ProfileDirtyApp(DirtyApp):
    """Dirty app that remembers which process ran ``init``."""

    # Explicit worker limit; combined with the "module:Class:1" spec it
    # also keeps the master from importing this module up-front.
    workers = 1

    def __init__(self):
        self.initialized = False
        self.init_pid = None

    def init(self):
        self.initialized = True
        self.init_pid = os.getpid()

    def where_am_i(self):
        return {
            "initialized": self.initialized,
            "init_pid": self.init_pid,
            "call_pid": os.getpid(),
        }

    def compute(self, a, b):
        return a + b

    def close(self):
        self.initialized = False
        self.init_pid = None
