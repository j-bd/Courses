import numpy as np

class Flag:
    def __init__(self, x):
        self.x = x


class TypeFloat(Flag):
    def flag(self):
        return isinstance(self.x, float)
        

class PositiveValue(Flag):
    def flag(self):
        try:
            return self.x > 0
        except TypeError:
            return np.nan

class Even(Flag):
    def flag(self):
        try:
            return self.x%2 == 0
        except TypeError:
            return np.nan

class Overall(Flag):
    FLAGS = [TypeFloat, PositiveValue, Even]

    def flag(self):
        results = []
        cls = self.__class__
        for f in cls.FLAGS:
            results.append(f(self.x).flag())
        return results

        


if __name__ == "__main__":
    elements = ["a", True, 1., -1, 4]
    for e in elements:
        print(e, TypeFloat(e).flag(), PositiveValue(e).flag())

    for e in elements:
        print(e, Overall(e).flag())
