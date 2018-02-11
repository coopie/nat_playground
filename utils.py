import importlib.util


def import_module(path):
    spec = importlib.util.spec_from_file_location('', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m
