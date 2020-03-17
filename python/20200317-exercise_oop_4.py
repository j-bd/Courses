import datetime
from dataclasses import dataclass
from functools import total_ordering

@dataclass
@total_ordering
class MyDatetime:
    birthday = datetime.datetime(1987, 11, 28)

    datetime: datetime.datetime

    def difference_to_birthday(self) -> int:
        cls = self.__class__
        diff = self.datetime - cls.birthday
        return int(diff.total_seconds())

    def __repr__(self):
        return str(self.datetime.timestamp())

    def __add__(self, i: int):
        new_datetime = self.datetime + datetime.timedelta(i)
        return MyDatetime(new_datetime)

    def __lt__(self, other):
        return self.datetime < other.datetime

    def __eq__(self, other):
        return self.datetime == other.datetime



@dataclass
class Flag:
    x: float

# pour comparer deux classes pour eviter de devoir
# implementer les 6 comparaison, on importe total_ordering
# on implement __eq__ et une autre au choix
