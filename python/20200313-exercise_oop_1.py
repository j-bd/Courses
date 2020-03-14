import datetime
from functools import total_ordering

@total_ordering
class MyDatetime:
    birthday = datetime.datetime(1987, 11, 28)

    def __init__(self, datetime):
        self.datetime = datetime

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



#bonus
class MyDatetime2:
    def __init__(self, datetime):
        self._datetime = datetime
        
    @property
    def datetime(self): #getter
        print("using getter")
        return self._datetime

    @datetime.setter
    def datetime(self, new_datetime):
        self._datetime = new_datetime
        print("using setter")
        return self._datetime


if __name__ == "__main__":
    today = datetime.datetime(2020, 1, 1)
    my_datetime = MyDatetime(today)
    diff = my_datetime.difference_to_birthday()
    print(diff)
    print(my_datetime + 2)

    yesterday = datetime.datetime(2019, 12, 31)
    my_datetime2 = MyDatetime(yesterday)
    print(my_datetime < my_datetime2)
    print(my_datetime == my_datetime2)
    print(my_datetime >= my_datetime2)

    my_datetime2 = MyDatetime2(today)
    print(my_datetime2.datetime)
    my_datetime2.datetime = yesterday
    print(my_datetime2.datetime)