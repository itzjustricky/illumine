"""
    Utilitiies to deal with time

"""

import datetime


class StopWatch(object):

    def __init__(self, session_name):
        self._session_name = session_name

    def __enter__(self):
        self.start_time = datetime.datetime.now()
        print("[{}] Entered session {}".format(
            self.start_time.strftime("%H:%M:%S"),
            self._session_name))

    def __exit__(self, *args):
        end_time = datetime.datetime.now()
        time_taken = end_time - self.start_time

        print("{} time taken for session {}."
              .format(time_taken, self._session_name))

        print("[{}] Exiting session {}".format(
            end_time.strftime("%H:%M:%S"),
            self._session_name))
