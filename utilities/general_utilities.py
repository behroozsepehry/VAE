import importlib.util


def import_from_path(path):
    spec = importlib.util.spec_from_file_location('module_spec', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def append_key_dict(d, k):
    return {k+kk: vv for kk, vv in d.items()}
