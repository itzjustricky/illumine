"""
    Description:
        Tools to help with class management

"""


class AccessMeta(type):
    """ MetaClass to control overriding of functions in Base Classes.
        Modified to search through all Base classes in the Inheritance Hierarchy.

    ..note:
        Methods set to be final will have an attribute __final set to the
        AccessMeta.__SENTINEL value.

        Credit to: Uzumaki, Noctis Skytower from StackOverFlow
    """
    __SENTINEL = object()

    def __new__(cls, name, bases, attrs):
        # Get all the base classes lower in the hierachy
        all_bases = {b
                     for base in bases
                     for b in base.mro()}
        final_methods = {key
                         for base in all_bases
                         for key, value in vars(base).items()
                         if callable(value) and cls.__is_final(value)}

        violations = [key for key in attrs if key in final_methods]
        if any(violations):
            raise RuntimeError(
                "{} method(s) declared final and may not be overridden".format(violations))
        return super().__new__(cls, name, bases, attrs)

    @classmethod
    def __is_final(cls, method):
        """  """
        try:
            return method.__final is cls.__SENTINEL
        except AttributeError:
            return False

    @classmethod
    def final(cls, method):
        """ Decorator to set a method to be final """
        method.__final = cls.__SENTINEL
        return method
