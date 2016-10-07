"""
    Module of utility methods to deal with functions

"""

import time
import logging
from functools import wraps


def timethis(func):
    """ Decorator that reports the execution time """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(func.__name__, end - start)
        return result
    return wrapper


# I need to take the time to understand this sometime
def logged(level, name=None, message=None):
    """ Logging decorator for a function

    :param level: the logging level
    :param name: the logger name
    :param message: is the log message; if name and message aren't specified,
        they default to the function's module and name.

        # Example uses
        @logged(logging.DEBUG)
        def add(x, y):
            return x + y

        @logged(logging.CRITICAL, 'example')
        def spam():
            print('Spam!')
    """
    def decorate(func):
        logname = name if name else func.__module__
        log = logging.getLogger(logname)
        logmsg = message if message else func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            log.log(level, logmsg)
            return func(*args, **kwargs)
        return wrapper
    return decorate


def static_var(**kwargs):
    """ A decorator function to allow for static variables in a function

        Example use:
            @static_var(counter=0)
            def foo():
                foo.counter += 1
                print "Counter is %d" % foo.counter
    """
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


@static_var(counter=0)
def logstamp():
    """ Returns a timestamp for the current time using time module
    :returns: (string) timestamp of HH:MM:SS
    """
    logstamp.counter += 1
    return "({}) [{}]".format(logstamp.counter, time.strftime('%D %H:%M:%S'))
