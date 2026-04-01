class Timer:
    list: float = 0
    load: float = 0
    backup: float = 0
    calc: float = 0
    apply: float = 0
    move: float = 0
    restore: float = 0
    activate: float = 0
    deactivate: float = 0

    @property
    def total(self):
        return round(self.activate + self.deactivate, 2)

    @property
    def summary(self):
        t = {}
        for k, v in self.__dict__.items():
            if v > 0.1:
                t[k] = round(v, 2)
        return t

    def clear(self, complete: bool = False):
        self.backup = 0
        self.calc = 0
        self.apply = 0
        self.move = 0
        self.restore = 0
        if complete:
            self.activate = 0
            self.deactivate = 0

    def add(self, name, t):
        self.__dict__[name] += t

    def __str__(self):
        return f'{self.__class__.__name__}({self.summary})'
