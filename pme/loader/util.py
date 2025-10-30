

class MultiprocessingWrapper:

    def __init__(self, func, unpack='dict', **kwargs):
        self.func = func
        self.kwargs = kwargs
        self.unpack = unpack

    def run(self, input):
        if self.unpack == 'list':
            return self.func(*input, **self.kwargs)
        elif self.unpack == 'dict':
            return self.func(**input, **self.kwargs)
        else:
            return self.func(input, **self.kwargs)