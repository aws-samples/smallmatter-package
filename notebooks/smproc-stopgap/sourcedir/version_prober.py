import importlib


def probe_ml_frameworks():
    pkgs = dict()
    for module_name in ["mxnet", "sklearn", "torch", "tensorflow", "xgboost"]:
        try:
            pkgs[module_name] = importlib.import_module(module_name)
        except ModuleNotFoundError:
            pass
    return pkgs
