import unittest
import pyrootutils

from state_tracking.python import PythonTracker


PROJECT_ROOT = path = pyrootutils.find_root(
    search_from=__file__, indicator=".project-root"
)

class TestData(unittest.TestCase):  # noqa: D101

    def test_toggle_tuple(self):
        tracker = PythonTracker.initialize(True, False)
        tracker.transpose(0, 1)
        program = [
            "x0 = True",
            "x1 = False",
            "x0, x1 = x1, x0",
        ]
        self.assertListEqual(tracker.history, program)

        locals_ = {}
        exec("\n".join(tracker.history), None, locals_)
        self.assertFalse(locals_["x0"])
        self.assertTrue(locals_["x1"])

    def test_toggle_z(self):
        tracker = PythonTracker.initialize(True, False, mode="z")
        tracker.transpose(0, 1)
        program = [
            "x0 = True",
            "x1 = False",
            "z = x0",
            "x0 = x1",
            "x1 = z",
        ]
        self.assertListEqual(tracker.history, program)

        locals_ = {}
        exec("\n".join(tracker.history), None, locals_)
        self.assertFalse(locals_["x0"])
        self.assertTrue(locals_["x1"])
