import glob
import os.path as osp
import pickle


class Bidict(dict):
    """
    A many-to-one bidirectional dictionary
    Reference: https://stackoverflow.com/questions/3318625/how-to-implement-an-efficient-bidirectional-hash-table
    """

    def __init__(self, *args, **kwargs):
        super(Bidict, self).__init__(*args, **kwargs)
        self.inverse = {}
        for key, value in self.items():
            self.inverse.setdefault(value, set()).add(key)

    def __setitem__(self, key, value):
        if key in self:
            self.inverse[self[key]].remove(key)
        super(Bidict, self).__setitem__(key, value)
        self.inverse.setdefault(value, set()).add(key)

    def __delitem__(self, key):
        self.inverse[self[key]].remove(key)
        if len(self.inverse[self[key]]) == 0:
            del self.inverse[self[key]]
        super(Bidict, self).__delitem__(key)


class OrderedBidict(dict):
    """
    A one-to-many bidirectional dictionary whose value is a list instead of a set
    """

    def __init__(self, *args, **kwargs):
        super(OrderedBidict, self).__init__(*args, **kwargs)
        self.inverse = {}
        for key, values in self.items():
            for value in values:
                assert value not in self.inverse  # no duplicates
                self.inverse[value] = key

    def __setitem__(self, key, value):
        raise NotImplementedError

    def __delitem__(self, key):
        raise NotImplementedError


class LazyDict(dict):
    def __init__(self, dir_cache, prefix):
        super().__init__()
        self.buffer = {}
        self.dir_cache = dir_cache
        self.path_prefix = osp.join(dir_cache, f"{prefix}_")
        self._keys = [
            self.removeprefix(x, self.path_prefix)
            for x in glob.glob(self.path_prefix + "*")
        ]

    def keys(self):
        return self._keys

    def values(self):
        return [self.__getitem__(key) for key in self._keys]

    def items(self):
        raise NotImplementedError

    def __getitem__(self, key):
        if key in self.buffer:
            return self.buffer[key]
        else:
            with open(self.path_prefix + key, "rb") as f:
                value = pickle.load(f)
                self.buffer[key] = value
                return value

    def __len__(self):
        return len(self._keys)

    def __repr__(self):
        return "LazyDict()"

    # Added for python backwards compatibility
    def removeprefix(self, text, prefix):
        if text.startswith(prefix):
            return text[len(prefix) :]
        return text
