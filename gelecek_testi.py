"""Nedensellik testi: bir hastanın son saatlerini silip dönüştürürsek, kalan saatlerin çıktısı DEĞİŞMEMELİ."""
import sys, runpy, numpy as np
sys.argv = ["onisleme.py", "A->B"] + sys.argv[1:]
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path("onisleme.py")
oku, donustur, bolme = g["oku"], g["donustur"], g["bolme"]
for hid in bolme.hasta_id.sample(20, random_state=0):
    df = oku(hid); k = len(df) // 2
    tam, _ = donustur(df); kesik, _ = donustur(df.iloc[:k].copy())
    assert np.allclose(tam[:k], kesik), f"GELECEK SIZINTISI: {hid}"
print("Nedensellik testi geçti ✔ (gelecek saatler geçmişi etkilemiyor)")
