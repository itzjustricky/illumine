"""
    Utilitiies to deal with time

"""

import datetime


class StopWatch(object):

    def __init__(self, session_name):
        self._session_name = session_name

    def __enter__(self):
        self.start_time = datetime.datetime.now()

    def __exit__(self, *args):
        time_taken = datetime.datetime.now() - self.start_time
        print("{} time taken for session {}."
              .format(time_taken, self._session_name))
