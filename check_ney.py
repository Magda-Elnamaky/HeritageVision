"""Run:  python check_net.py   and send me the output."""
import time
import requests

UA = {"User-Agent": "HeritageVision/1.0 (student project)"}
TESTS = [
    ("Nominatim", "get", "https://nominatim.openstreetmap.org/search", {"q": "Cairo", "format": "json", "limit": 1}),
    ("Open-Meteo geocoding", "get", "https://geocoding-api.open-meteo.com/v1/search", {"name": "Cairo", "count": 1}),
    ("Overpass (main)", "post", "https://overpass-api.de/api/interpreter", {"data": "[out:json][timeout:10];node(30.04,31.23,30.05,31.24)[shop];out 1;"}),
    ("Overpass (kumi)", "post", "https://overpass.kumi.systems/api/interpreter", {"data": "[out:json][timeout:10];node(30.04,31.23,30.05,31.24)[shop];out 1;"}),
    ("Overpass (private.coffee)", "post", "https://overpass.private.coffee/api/interpreter", {"data": "[out:json][timeout:10];node(30.04,31.23,30.05,31.24)[shop];out 1;"}),
    ("Overpass (openstreetmap.fr)", "post", "https://overpass.openstreetmap.fr/api/interpreter", {"data": "[out:json][timeout:10];node(30.04,31.23,30.05,31.24)[shop];out 1;"}),
]
for name, how, url, data in TESTS:
    t = time.time()
    try:
        if how == "get":
            r = requests.get(url, params=data, headers=UA, timeout=(10, 20))
        else:
            r = requests.post(url, data=data, headers=UA, timeout=(10, 20))
        print(f"{name:30} HTTP {r.status_code}  {time.time() - t:.1f}s")
    except Exception as e:
        print(f"{name:30} FAILED  {time.time() - t:.1f}s  {type(e).__name__}: {str(e)[:120]}")