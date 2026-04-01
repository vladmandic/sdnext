from modules import scripts_postprocessing, ui_sections, processing_grading


class ScriptPostprocessingColorGrading(scripts_postprocessing.ScriptPostprocessing):
    name = "Color Grading"

    def ui(self):
        ui_controls = ui_sections.create_color_inputs('process')
        ui_controls_dict = {control.label.replace(" ", "_").replace(".", "").lower(): control for control in ui_controls}
        return ui_controls_dict

    def process(self, pp: scripts_postprocessing.PostprocessedImage, *args, **kwargs): # pylint: disable=arguments-differ
        grading_params = processing_grading.GradingParams(*args, **kwargs)
        if processing_grading.is_active(grading_params):
            pp.image = processing_grading.grade_image(pp.image, grading_params)
