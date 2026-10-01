# Shim minimo di pytest (fixture, mark.parametrize, raises, approx) per far girare i test unitari della PR senza
# installare nulla nel venv di produzione. Solo per i test del backport #57553.
import contextlib, itertools

_FIXTURES = {}


def fixture(fn=None, **_):
    def deco(f):
        _FIXTURES[f.__name__] = f
        return f
    return deco(fn) if fn else deco


class _Mark:
    def parametrize(self, names, values, **_):
        names = [n.strip() for n in names.split(",")] if isinstance(names, str) else list(names)
        def deco(f):
            f.__dict__.setdefault("_params", []).append((names, list(values)))
            return f
        return deco

    def __getattr__(self, _):
        return lambda *a, **k: (lambda f: f)


mark = _Mark()


@contextlib.contextmanager
def raises(exc, match=None):
    try:
        yield
    except exc:
        return
    raise AssertionError(f"{exc} non sollevata")


def param(*values, **_):
    return values if len(values) > 1 else values[0]


def run_module(mod):
    import inspect
    ok = fail = 0
    items = [(n, f) for n, f in vars(mod).items() if n.startswith("test_") and callable(f)]
    for cname, cls in vars(mod).items():
        if cname.startswith("Test") and isinstance(cls, type):
            inst = cls()
            items += [(f"{cname}.{n}", getattr(inst, n)) for n in dir(cls) if n.startswith("test_")]
    for name, fn in items:
        groups = getattr(fn, "__func__", fn).__dict__.get("_params", [])
        combos = [{}]
        for names, values in groups:
            combos = [dict(c, **dict(zip(names, v if len(names) > 1 else (v,)))) for c in combos for v in values]
        for combo in combos:
            def resolve(arg, cache):
                if arg in combo: return combo[arg]
                if arg not in cache:
                    f = _FIXTURES[arg]
                    cache[arg] = f(**{a: resolve(a, cache) for a in inspect.signature(f).parameters})
                return cache[arg]
            cache = {}
            try:
                fn(**{a: resolve(a, cache) for a in inspect.signature(fn).parameters})
                ok += 1
            except Exception as e:  # noqa: BLE001
                fail += 1
                print(f"FAIL {name} {combo}: {type(e).__name__}: {str(e)[:200]}")
    print(f"{mod.__name__}: {ok} ok, {fail} fail")
    return fail
