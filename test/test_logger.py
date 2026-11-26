import logging
import unittest
from contextlib import nullcontext
from io import StringIO
from unittest.mock import patch

from rich.console import Console
from tqdm import tqdm

from modules.logger import TqdmRichHandler


class TestTqdmRichHandler(unittest.TestCase):
    def test_emit_does_not_create_lock_when_no_bars_are_active(self):
        output = StringIO()
        handler = TqdmRichHandler(show_time=False, console=Console(file=output))
        record = logging.LogRecord('test', logging.INFO, '', 0, 'message', (), None)

        with patch.object(tqdm, '_lock', None, create=True), \
                patch.object(tqdm, '_instances', []), \
                patch('tqdm.tqdm.get_lock') as get_lock, \
                patch('tqdm.tqdm.external_write_mode') as external_write_mode:
            handler.emit(record)

        get_lock.assert_not_called()
        external_write_mode.assert_not_called()
        self.assertIn('message', output.getvalue())

    def test_emit_uses_external_write_mode_when_bars_are_active(self):
        output = StringIO()
        handler = TqdmRichHandler(show_time=False, console=Console(file=output))
        record = logging.LogRecord('test', logging.INFO, '', 0, 'message', (), None)

        with patch.object(tqdm, '_instances', [object()]), \
                patch('tqdm.tqdm.external_write_mode', return_value=nullcontext()) as external_write_mode:
            handler.emit(record)

        external_write_mode.assert_called_once_with(file=output, nolock=True)
        self.assertIn('message', output.getvalue())


if __name__ == '__main__':
    unittest.main()
