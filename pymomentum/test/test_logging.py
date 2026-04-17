# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import pymomentum.geometry as pym_geometry  # @manual=:geometry


class TestLogging(unittest.TestCase):
    def test_log_level_roundtrip(self) -> None:
        """Verify set_log_level/get_log_level round-trip correctly."""
        original = pym_geometry.get_log_level()
        for level in [
            pym_geometry.LogLevel.Disabled,
            pym_geometry.LogLevel.Error,
            pym_geometry.LogLevel.Warning,
            pym_geometry.LogLevel.Info,
            pym_geometry.LogLevel.Debug,
            pym_geometry.LogLevel.Trace,
        ]:
            pym_geometry.set_log_level(level)
            self.assertEqual(pym_geometry.get_log_level(), level)
        # Restore original level:
        pym_geometry.set_log_level(original)

    def test_set_log_level_by_string(self) -> None:
        """Verify set_log_level accepts string names."""
        original = pym_geometry.get_log_level()
        pym_geometry.set_log_level("Warning")
        self.assertEqual(pym_geometry.get_log_level(), pym_geometry.LogLevel.Warning)
        pym_geometry.set_log_level(original)

    def test_redirect_logs_to_python(self) -> None:
        """Verify redirect_logs_to_python returns a bool without crashing."""
        result = pym_geometry.redirect_logs_to_python()
        self.assertIsInstance(result, bool)

    def test_multithreaded_logging_does_not_crash(self) -> None:
        """Verify that MT_LOGW from multiple C++ threads does not crash.

        This exercises the pythonLogCallback GIL acquisition path: if the
        redirect is active (XR_LOGGER build), each C++ thread must acquire
        the GIL before calling py::print.  Without correct GIL handling
        this would segfault or deadlock.
        """
        # Activate the Python log redirect (no-op if already active or if
        # XR_LOGGER is not available):
        pym_geometry.redirect_logs_to_python()
        pym_geometry.set_log_level(pym_geometry.LogLevel.Warning)

        # Fire MT_LOGW from 8 threads, 50 messages each.  The function
        # releases the GIL while waiting for threads, so the callbacks can
        # re-acquire it without deadlocking.
        pym_geometry._test_multithreaded_logging(
            num_threads=8, num_messages_per_thread=50
        )

        # If we get here without crashing, the test passes.
