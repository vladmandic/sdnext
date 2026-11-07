import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi.exceptions import HTTPException

from modules import scripts_manager
from modules.api import control as control_api


class TestAPIControlScripts(unittest.TestCase):
    class Script:
        def __init__(self, name, alwayson=True, args_from=1, args_to=3):
            self.name = name
            self.alwayson = alwayson
            self.args_from = args_from
            self.args_to = args_to

        def title(self):
            return self.name

    def _runner(self, *scripts):
        return SimpleNamespace(scripts=list(scripts), initialize_scripts=Mock())

    def test_post_control_passes_named_alwayson_args_and_preserves_selectable_args(self):
        api = control_api.APIControl(threading.Lock())
        api.default_script_arg = [0, False, 0.5]
        runner = self._runner(self.Script('NudeNet'))
        request = control_api.ReqControl(
            script_name='selectable',
            script_args=[42],
            alwayson_scripts={'NudeNet': {'args': [True, 0.8]}},
        )

        with patch.object(scripts_manager, 'scripts_control', runner), \
                patch.object(api, 'prepare_face_module'), \
                patch.object(api, 'prepare_control'), \
                patch.object(api, 'prepare_xyz_grid'), \
                patch.object(api, 'prepare_ip_adapter', return_value={}), \
                patch.object(control_api.run, 'control_set') as set_control_args, \
                patch.object(control_api.run, 'control_run', return_value=[]) as run_control, \
                patch.object(control_api.shared.mem_mon, 'reset'), \
                patch.object(control_api.shared.state, 'begin', return_value='job'), \
                patch.object(control_api.shared.state, 'end'), \
                patch.object(control_api.shared.state, 'error', None), \
                patch('modules.ui_control.return_stats', return_value=''):
            api.post_control(request)

        self.assertEqual(set_control_args.call_args.args[0]['per_script_args'], {'NudeNet': [True, 0.8]})
        self.assertNotIn('alwayson_scripts', run_control.call_args.kwargs)
        self.assertEqual(run_control.call_args.kwargs['override_script_name'], 'selectable')
        self.assertEqual(run_control.call_args.kwargs['override_script_args'], [42])

    def test_missing_alwayson_values_use_defaults_without_named_overrides(self):
        api = control_api.APIControl(threading.Lock())
        api.default_script_arg = [0, False, 0.5]
        runner = self._runner(self.Script('NudeNet'))

        with patch.object(scripts_manager, 'scripts_control', runner):
            overrides = api.prepare_scripts(control_api.ReqControl())

        self.assertEqual(overrides, {})

    def test_unknown_and_selectable_scripts_are_rejected(self):
        api = control_api.APIControl(threading.Lock())
        api.default_script_arg = [0, False, 0.5]
        cases = (
            (self._runner(self.Script('NudeNet')), 'missing'),
            (self._runner(self.Script('Selectable', alwayson=False)), 'Selectable'),
        )

        for runner, name in cases:
            with self.subTest(script=name), patch.object(scripts_manager, 'scripts_control', runner):
                request = control_api.ReqControl(alwayson_scripts={name: {'args': [True, 0.8]}})
                with self.assertRaises(HTTPException) as error:
                    api.prepare_scripts(request)
                self.assertEqual(error.exception.status_code, 422)


if __name__ == '__main__':
    unittest.main()
