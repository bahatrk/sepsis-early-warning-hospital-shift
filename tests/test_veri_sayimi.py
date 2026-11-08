import importlib.util, pathlib
spec = importlib.util.spec_from_file_location("sayim", pathlib.Path(__file__).parents[1] / "scripts" / "00_veri_sayimi.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def _psv(p, labels):
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = ["HR|SepsisLabel"] + [f"{70+i if i%2==0 else ''}|{l}" for i, l in enumerate(labels)]
    p.write_text("\n".join(rows))

def test_sayim(tmp_path):
    _psv(tmp_path/"training_setA"/"p000001.psv", [0, 0, 1, 1])
    _psv(tmp_path/"training_setA"/"p000002.psv", [0, 0, 0])
    _psv(tmp_path/"training_setB"/"p100001.psv", [0, 1])
    s = m.say(str(tmp_path))
    assert s["A"]["hasta"] == 2 and s["A"]["satir"] == 7 and s["A"]["sepsisli_hasta"] == 1 and s["A"]["pozitif_satir"] == 2
    assert s["B"]["hasta"] == 1 and s["toplam"]["hasta"] == 3 and s["toplam"]["satir"] == 9
    assert abs(s["A"]["eksik_orani"]["HR"] - 3/7) < 1e-3
