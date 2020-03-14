import datetime
import functools

def log_output(func):
    def wrapper(*args, **kwargs):
        output = func(*args, **kwargs)
        LOG_FILE = "output.txt"
        with open(LOG_FILE, "w") as text_file:
            text_file.write(str(output))
        return output
    return wrapper


def log_output2(output_file):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            output = func(*args, **kwargs)
            with open(output_file, "w") as text_file:
                text_file.write(str(output))
            return output
        return wrapper
    return decorator


class MyDatetime:
    birthday = datetime.datetime(1987, 11, 28)

    def __init__(self, datetime):
        self.datetime = datetime

    @log_output2("test.txt") #or @log_output
    def difference_to_birthday(self) -> int:
        cls = self.__class__
        diff = self.datetime - cls.birthday
        return int(diff.total_seconds())

    @classmethod
    def age(cls):
        now = datetime.datetime.now()
        diff = now - cls.birthday
        N_DAYS_IN_YEAR = 365.25
        age_in_years = diff.days / N_DAYS_IN_YEAR #not 100% exact but that's not the point
        return int(age_in_years) #int acts as floor

    @property
    def is_christmas(self):
        is_day = self.datetime.day == 25
        is_month = self.datetime.month == 12
        return is_day and is_month

    @staticmethod
    def unix_epoch_time():
        now = datetime.datetime.now()
        return now.timestamp()

if __name__ == "__main__":
    today = datetime.datetime(2017, 1, 1)
    print(MyDatetime(today).difference_to_birthday())
    print(MyDatetime.age())

    christmas = datetime.datetime(2020, 12, 25)
    print(MyDatetime(christmas).is_christmas)
    not_christmas = datetime.datetime(2020, 12, 24)
    print(MyDatetime(not_christmas).is_christmas)

    print(MyDatetime.unix_epoch_time())