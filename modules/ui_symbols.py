import re
from functools import lru_cache
from typing import final


# Basic symbols

refresh = '⟲'
close = '✕'
load = '⇧'
save = '⇩'
book = '🕮'
apply = '⇰'
clear = '⊗'
fill = '⊜'
scan = '🔎︎'
view = '☲'
networks = '🌐'
paste = '⇦'
refine = '※'
switch = '⇅'
sort = '⇕'
detect = '📐'
folder = '📂'
random = '🎲️'
reuse = '♻️'
info = 'ℹ' # noqa
reset = '🔄'
upload = '⬆️'
loading = '↺'
reuse = '⬅️'
search = '🔍'
preview = '🖼️'
image = '🖌️'
resize = '⁜'
caption = '\uf46b' # Telescope icon in Noto Sans. Previously '♻'
bullet = '⃝'
vision = '\uf06e'  # Font Awesome eye icon (more minimalistic)
reasoning = '\uf0eb'  # Font Awesome lightbulb icon (represents thinking/reasoning)
sort_alpha_asc = '\uf15d'
sort_alpha_dsc = '\uf15e'
sort_size_asc = '\uf160'
sort_size_dsc = '\uf161'
sort_num_asc = '\uf162'
sort_num_dsc = '\uf163'
sort_time_asc = '\uf0de'
sort_time_dsc = '\uf0dd'
style_apply = '↶'
style_save = '↷'

# Configurable symbols

@final
class SVGSymbol:
    __created = []
    __re_display = re.compile(r"(?<=display:)\s*([\w\-]+)(?=;)")

    @classmethod
    @lru_cache  # Class method due to B019, but also mostly so the `style` method shows params in IDE
    def __stylize(cls, svg: str, color: str | None = None, display: str | None = None):
        if color:
            svg = re.sub("currentColor", color, svg)
        if display:
            svg = cls.__re_display.sub(display, svg, count=1)
        return svg

    def __init__(self, svg: str):
        svg = re.sub(r"\s{2,}", " ", svg.replace("\n", "")).replace("> <", "><").strip()
        if svg in self.__created:
            raise RuntimeError("SVGSymbol class was created with an existing value. There should only be one instance per symbol.", svg)
        else:
            self.__created.append(svg)
        self.svg = svg
        self.supports_color = False
        self.supports_display = False
        if "currentColor" in self.svg:
            self.supports_color = True
        if self.__re_display.search(self.svg):
            self.supports_display = True

    def style(self, color: str | None = None, display: str | None = None) -> str:
        style_args = {
            "color": color if color and self.supports_color else None,
            "display": display if display and self.supports_display else None
        }
        return self.__stylize(self.svg, **style_args)

    def __str__(self):
        return self.svg


svg_bullet = SVGSymbol("<svg style='stroke:currentColor;fill:none;stroke-width:2;display:block;' viewBox='0 0 16 16'><circle cx='8' cy='8' r='7'/></svg>")
