from functools import partial

class domain(list):
    def __init__(self, **kwargs):
        self.n = 1
        self.place_values = {}
        self.place_moduli = {}
        self.keys = list(kwargs.keys())

        for k, v in reversed(kwargs.items()):
            assert isinstance(v, int)
            assert v > 1
            assert not k.startswith('_')
            self.place_values[k] = self.n
            self.n *= v
            self.place_moduli[k] = v

        for k in kwargs.keys():
            self.__setattr__(k, partial(self.unpack, k=k))

    def _update(self, z, **kwargs):
        for k in kwargs.keys():
            assert k in self.keys

        kwargs_ = {k: kwargs.get(k, self.unpack(z, k)) for k in self.keys}
        return self.pack(**kwargs_)

    def _tuple(self, z):
        return tuple(self.unpack(z, k) for k in self.keys)

    def __iter__(self):  # fool JAX into thinking I'm an array
        return iter(range(self.n))

    def __len__(self):
        return self.n

    def __call__(self, *args, **kwargs):
        return self.pack(*args, **kwargs)

    def pack(self, *args, **kwargs):
        z = 0
        if len(args) > 0:
            assert len(args) == len(self.keys)
            for i, v in enumerate(args):
                z = z + v * self.place_values[self.keys[i]]

        else:
            assert len(kwargs) == len(self.keys)
            for j, (k, v) in enumerate(kwargs.items()):
                assert self.keys[j + len(args)] == k
                z = z + v * self.place_values[k]

        return z

    def unpack(self, z, k):
        assert k in self.keys
        return (z // self.place_values[k]) % self.place_moduli[k]


import importlib.abc, importlib.util

class StringLoader(importlib.abc.SourceLoader):
    def __init__(self, data):
        self.data = data
    def get_source(self, fullname):
        return self.data
    def get_data(self, path):
        return self.data.encode("utf-8")
    def get_filename(self, fullname):
        return "<dynamic>"

def make_module(name):
    loader = StringLoader('''
def install(x):
    exec(x, globals())
    return globals()
    ''')
    spec = importlib.util.spec_from_loader(name, loader, origin="built-in")
    module = importlib.util.module_from_spec(spec)
    # import sys
    # sys.modules[name] = module
    spec.loader.exec_module(module)
    return module