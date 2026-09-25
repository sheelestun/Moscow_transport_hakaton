"""Focused regression tests for temporal leakage and diagnostic semantics."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze import point_coverage, quality_flags


class TemporalCoverageTests(unittest.TestCase):
    def setUp(self):
        self.now = pd.Timestamp("2026-01-06 12:00:00")
        self.points = pd.DataFrame({"tr_id": [1, 2], "T": [self.now, self.now]})

    def traffic(self, offsets, valid=None):
        return pd.DataFrame({"tr_id": [1]*len(offsets), "event_time": [self.now+pd.Timedelta(x, unit="ns") for x in offsets], "location_valid": valid or [True]*len(offsets), "lon": [37.0]*len(offsets), "lat": [55.0]*len(offsets)})

    def test_open_left_closed_right_and_nanosecond_future(self):
        t = self.traffic([-900_000_000_000, -1, 0, 1])
        out = point_coverage(self.points, t)
        self.assertEqual(out.loc[0, "valid_pings_15m"], 2)
        self.assertEqual(out.loc[0, "valid_ping_age_s"], 0)
        self.assertTrue(np.isnan(out.loc[1, "valid_ping_age_s"]))
        self.assertEqual(out.loc[1, "valid_pings_15m"], 0)

    def test_future_and_input_order_do_not_change_results(self):
        t = self.traffic([-10_000_000_000, -5_000_000_000])
        before = point_coverage(self.points, t)
        future = self.traffic([1, 100_000_000_000])
        after = point_coverage(self.points, pd.concat([future, t]).sample(frac=1, random_state=3))
        pd.testing.assert_frame_equal(before, after)

    def test_invalid_and_missing_coordinates_are_not_usable(self):
        t = self.traffic([-30_000_000_000, -10_000_000_000, 0], [True, False, True])
        t.loc[2, "lon"] = np.nan
        out = point_coverage(self.points, t)
        self.assertEqual(out.loc[0, "valid_ping_age_s"], 30)
        self.assertEqual(out.loc[0, "valid_pings_15m"], 1)

    def test_old_ping_is_not_fresh_but_history_exists(self):
        out = point_coverage(self.points, self.traffic([-901_000_000_000]))
        self.assertEqual(out.loc[0, "valid_ping_age_s"], 901)
        self.assertEqual(out.loc[0, "valid_pings_15m"], 0)


class QualityTests(unittest.TestCase):
    def test_flags_overlap_and_speed_threshold_is_strict(self):
        now = pd.Timestamp("2026-01-01")
        t = pd.DataFrame({"location_valid": [False, True], "gps_time": [now+pd.Timedelta(seconds=1), now], "lon": [37., 37.], "lat": [55., 55.], "alt": [np.nan, 1.], "speed": [121., 120.], "heading": [0., 0.], "event_time": [now, now], "receive_time": [now-pd.Timedelta(seconds=1), now]})
        flags = quality_flags(t)
        self.assertTrue(flags.iloc[0].all())
        self.assertFalse(flags.iloc[1].any())
        self.assertEqual(flags.value_counts().sum(), len(t))


if __name__ == "__main__":
    unittest.main()
