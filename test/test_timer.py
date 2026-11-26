import unittest

from modules.timer import Timer


class TestTimer(unittest.TestCase):
    def test_aggregate_records_are_excluded_from_total(self):
        timer = Timer()
        timer.add('phase', 2)
        timer.add('aggregate', 5, include_total=False)

        self.assertEqual(timer.get_total(), 2)
        self.assertIn('aggregate=5.00', timer.summary(min_time=0))
        self.assertEqual(timer.total, 2)

    def test_merge_preserves_excluded_records_and_total(self):
        timer = Timer()
        timer.add('phase', 2)
        other = Timer()
        other.add('aggregate', 5, include_total=False)

        timer.merge(other)

        self.assertEqual(timer.get_total(), 2)
        self.assertEqual(timer.records, {'phase': 2, 'aggregate': 5})

    def test_dct_does_not_mutate_records_with_synthetic_total(self):
        timer = Timer()
        timer.add('phase', 2)

        self.assertEqual(timer.dct(), {'phase': 2, 'total': 2})
        self.assertEqual(timer.records, {'phase': 2})

    def test_explicit_total_overrides_overlapping_breakdown(self):
        timer = Timer()
        timer.add('process', 12.7, include_total=False)
        timer.add('pipeline', 10.61)
        timer.add('wall', 12.75, include_total=False)
        timer.set_total(12.75)

        self.assertEqual(timer.get_total(), 12.75)
        self.assertEqual(timer.dct()['total'], 12.75)
        self.assertEqual(timer.get('pipeline'), 10.61)


if __name__ == '__main__':
    unittest.main()
