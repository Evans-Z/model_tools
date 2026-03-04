class DummyModel:
    def __call__(self, *args, **kwargs):
        return {"args": args, "kwargs": kwargs}


def build_dict_case():
    return {
        "model": DummyModel(),
        "inputs": {"x": 1},
        "kwargs": {"y": 2},
        "metadata": {"name": "dict_case"},
        "source_path": __file__,
        "entry_class": "DummyModel",
    }


def build_tuple_case():
    return DummyModel(), {"x": 1}


def build_tuple3_case():
    return DummyModel(), (1, 2), {"z": 3}
