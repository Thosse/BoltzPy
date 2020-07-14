import numpy as np


class BaseClass:
    def __eq__(self, other, ignore=None,print_message=True):
        if ignore is None:
            ignore = []
        # This explicitly allows other to be an child class of self
        if not isinstance(other, type(self)):
            if print_message:
                print("Objects are of different type:",
                      "\n\ttype(self) = ", type(self),
                      "\n\ttype(other) = ", type(other))
            return False
        if set(self.__dict__.keys()) != set(other.__dict__.keys()):
            if print_message:
                print("Objects have different attributes:",
                      "\n\tself.keys = ", set(self.__dict__.keys()),
                      "\n\tother.keys = ", set(other.__dict__.keys()))
            return False
        for (key, value) in self.__dict__.items():
            if key in ignore:
                continue
            other_value = other.__dict__[key]
            if type(value) != type(other_value):
                if print_message:
                    print("An attribute is of different type:",
                          "\n\tAttribute = ", key,
                          "\n\ttype(self) = ", type(value),
                          "\n\ttype(other) = ", type(other_value))
                return False
            if isinstance(value, np.ndarray):
                if value.shape != other_value.shape:
                    if print_message:
                        print("An attribute has differing shapes:",
                              "\n\tAttribute = ", key,
                              "\n\tself.attr.shape = ", value.shape,
                              "\n\tother.attr.shape) = ", other_value.shape)
                    return False
                if np.any(value != other_value):
                    if print_message:
                        print("An attribute has differing values:",
                              "\n\tAttribute = ", key,
                              "\n\tself.attr = ", value,
                              "\n\tother.attr = ", other_value)
                    return False
            else:
                if value != other_value:
                    if print_message:
                        print("An attribute has differing values:",
                              "\n\tAttribute = ", key,
                              "\n\ttype(self) = ", value,
                              "\n\ttype(other) = ", other_value)
                    return False
        return True
