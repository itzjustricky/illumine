"""
    Utility decorator functions

"""


def static_var(**kwargs):
    """ A decorator function to allow for static variables
        in a function

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
