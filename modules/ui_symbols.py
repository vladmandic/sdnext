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
interrogate = '\uf46b' # Telescope icon in Noto Sans. Previously '♻'
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

class SVGSymbol:
    def __init__(self, svg: str):
        self.svg = svg
        self.before = ""
        self.after = ""
        self.supports_color = False
        if "currentColor" in self.svg:
            self.supports_color = True
            self.before, self.after = self.svg.split("currentColor", maxsplit=1)

    def color(self, color: str):
        if self.supports_color:
            return self.before + color + self.after
        else:
            return self.svg

    def __str__(self):
        return self.svg

svg_bullet = SVGSymbol("<svg style='stroke:currentColor;fill:none;stroke-width:2;' viewBox='0 0 16 16'><circle cx='8' cy='8' r='7'/></svg>")
