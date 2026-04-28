import os
from dataclasses import fields
from modules import scripts_postprocessing, ui_sections, processing_grading


class ScriptPostprocessingColorGrading(scripts_postprocessing.ScriptPostprocessing):
    name = "Color Grading"

    def ui(self):
        ui_controls = ui_sections.create_color_inputs('process')
        ui_controls_dict = {control.label.replace(" ", "_").replace(".", "").lower(): control for control in ui_controls}
        return ui_controls_dict

    def process(self, pp: scripts_postprocessing.PostprocessedImage, *args, **kwargs): # pylint: disable=arguments-differ
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        grading_params = processing_grading.GradingParams(*args, **kwargs)
        if not processing_grading.is_active(grading_params):
            return
        pp.image = processing_grading.grade_image(pp.image, grading_params)
        defaults = processing_grading.GradingParams()
        for f in fields(grading_params):
            val = getattr(grading_params, f.name)
            if val == getattr(defaults, f.name):
                continue
            if f.name == 'lut_cube_file' and val:
                val = os.path.basename(str(val))
            pp.info[f"Grading {f.name.replace('_', ' ')}"] = val
