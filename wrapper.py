import time
import datetime

def run_time_record(logging):
    def _run_time_record(function):
        def inner(*args, **kwargs):
            start = time.time()
            ans = function(*args, **kwargs)
            print("%s %s(%s seconds)" % (
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), logging, round(time.time() - start, 3)))
            return ans
        return inner
    return _run_time_record

