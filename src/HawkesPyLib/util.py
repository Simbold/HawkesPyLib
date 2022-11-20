from abc import ABC, abstractmethod
import numpy as np


class Validator(ABC):
    """
    Abstract validator class
    """

    def __set_name__(self, owner, name):
        self.private_name = '_' + name

    def __get__(self, obj, objtype=None):
        return getattr(obj, self.private_name)

    def __set__(self, obj, value):
        value = self.validate(value)
        setattr(obj, self.private_name, value)

    @abstractmethod
    def validate(self, value):
        pass


class OneOf(Validator):
    """
    Validator Class which validates if string is part of multiple options
    """

    def __init__(self, *options):
        self.options = set(options)

    def validate(self, value):
        if value not in self.options:
            raise ValueError(f'Expected {value!r} to be one of {self.options!r}')

        return value


class FloatInExRange(Validator):
    """
    Validator class that validates if a given value is of type float or int and in the the exclusive range:
    lower_bound < value < upper_bound
    if the value is an int it will be converted to float
    """

    def __init__(self, param_name, lower_bound=None, upper_bound=None):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.param_name = param_name

    def validate(self, value):
        if not isinstance(value, (float, int)):
            raise TypeError(f'Expected {value!r} to be a float for parameter {self.param_name}')
        if self.lower_bound is not None and value <= self.lower_bound:
            raise ValueError(f'Expected {value!r} to be larger than {self.lower_bound!r} for parameter {self.param_name}')
        if self.upper_bound is not None and value >= self.upper_bound:
            raise ValueError(f'Expected {value!r} to be smaller than {self.upper_bound!r} for parameter {self.param_name}')
        return float(value)


class IntInExRange(Validator):
    """
    Validator class that validates if a given value is of type int and in the the exclusive range: lower_bound < value < upper_bound
    """

    def __init__(self, param_name, lower_bound=None, upper_bound=None):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.param_name = param_name

    def validate(self, value):
        if not isinstance(value, (int)):
            raise TypeError(f'Expected {value!r} to be a float')
        if self.lower_bound is not None and value <= self.lower_bound:
            raise ValueError(f'Expected {value!r} to be larger than {self.lower_bound!r} for parameter {self.param_name}')
        if self.upper_bound is not None and value >= self.upper_bound:
            raise ValueError(f'Expected {value!r} to be smaller than {self.upper_bound!r} for parameter {self.param_name}')
        return value


class PositiveFloatNdarray(Validator):
    """
    Validator class that validates if given value is a positive float numpy.ndarray
    """

    def __init__(self, param_name):
        self.param_name = param_name

    def validate(self, np_arr):
        if not isinstance(np_arr, (np.ndarray)):
            raise TypeError(f'Expected {np_arr!r} to be a np.ndarray for parameter {self.param_name}')
        if (np_arr <= 0).any():
            raise ValueError(f'Expected {np_arr!r} to contain only positive values for parameter {self.param_name}')
        if np_arr.dtype != np.float64:
            raise TypeError(f'Expected {np_arr!r} to be of dtype np.float64 for parameter {self.param_name}')
        return np_arr


class PositiveOrderedFloatNdarray(Validator):
    """
    Validator class that validates if given array contains only positive floats and is in acending order
    """

    def __init__(self, name):
        self.name = name

    def validate(self, np_arr):
        if not isinstance(np_arr, np.ndarray):
            raise TypeError(f'Expected the array {np_arr!r} to be a np.ndarray input argument {self.name}')
        if (np_arr.size == 0):
            raise ValueError(f"The array {np_arr!r} cannot be empty for input argument {self.name}")
        if (np_arr <= 0).any():
            raise ValueError(f'Expected the array {np_arr!r} to contain only positive values for input argument {self.name}')
        if np_arr.dtype != np.float64:
            raise TypeError(f'Expected the array {np_arr!r} to be of dtype np.float64 for input argument {self.name}')
        if not (np.diff(np_arr) >= 0).all():
            raise ValueError(f"The array {np_arr!r} must be sorted in acending order for input argument {self.name}")
        return np_arr
