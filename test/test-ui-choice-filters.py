"""CPU-only regression coverage for display-only sampler and upscaler filters."""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from modules.ui_choices import filter_ui_choices


class TestUiChoiceFilters(unittest.TestCase):
    def setUp(self):
        self.choices = ['Default', 'Euler', 'DPM++ 2M', 'Lanczos']

    def test_empty_preferences_leave_choices_unfiltered(self):
        visible, filtered = filter_ui_choices(self.choices, [])
        self.assertEqual(visible, self.choices)
        self.assertFalse(filtered)

    def test_preferences_filter_only_current_catalog_choices(self):
        visible, filtered = filter_ui_choices(self.choices, ['Euler', 'Lanczos'])
        self.assertEqual(visible, ['Euler', 'Lanczos'])
        self.assertTrue(filtered)

    def test_stale_preferences_do_not_hide_the_catalog(self):
        visible, filtered = filter_ui_choices(self.choices, ['Removed sampler'])
        self.assertEqual(visible, self.choices)
        self.assertFalse(filtered)

    def test_saved_selection_remains_available_when_not_preferred(self):
        visible, filtered = filter_ui_choices(self.choices, ['Euler'], selected='DPM++ 2M')
        self.assertEqual(visible, ['Euler', 'DPM++ 2M'])
        self.assertTrue(filtered)


if __name__ == '__main__':
    unittest.main()
