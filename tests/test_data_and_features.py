import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

import config
from data.clean import clean_hourly, to_daily
from data.download_siata import parse_geoportal_stations, parse_historical, parse_series_72h
from models.evaluate import blocked_time_folds, fast_f1_macro, mcnemar_exact
from models.features import build_features


@pytest.mark.parametrize("v,c", [(0, 0), (37.0, 0), (37.4, 0), (37.5, 1), (55.4, 1), (55.5, 2), (120, 2)])
def test_class_thresholds(v, c):
    assert config.pm25_to_class(v) == c


def test_clean_rules_and_75pct_rule():
    hours = pd.date_range("2024-01-01", periods=48, freq="h").strftime("%Y-%m-%d %H:%M:%S")
    vals = [20.0] * 48
    vals[0], vals[1], vals[2] = -9999, 900, -3          # faltante, implausible, negativo
    cal = [1.0] * 48
    cal[3] = 3.0                                          # bandera de calidad inválida
    for i in range(24, 31):                               # día 2: 7 h inválidas -> 17 válidas
        vals[i] = -9999
    raw = pd.DataFrame({"codigo": 1, "fecha": hours, "pm25": vals, "calidad": cal})
    h = clean_hourly(raw)
    assert h["pm25"].isna().sum() == 4 + 7
    d = to_daily(h)
    assert list(d["fecha"].dt.day) == [1]                 # día 2 descartado (<18 h)
    assert d["horas_validas"].iloc[0] == 20 and d["clase_nombre"].iloc[0] == "Normal"


def test_parsers():
    hist, st = parse_historical([{"codigoSerial": 7, "nombre": "X", "latitud": 6.2, "longitud": -75.5,
                                  "datos": [{"fecha": "2024-01-01 00:00:00", "valor": 12, "calidad": 1}]}])
    assert hist.iloc[0].tolist()[:3] == [7, "2024-01-01 00:00:00", 12.0] and st.iloc[0]["latitud"] == 6.2
    geo = parse_geoportal_stations({"features": [{"properties": {"codigo": 7, "nombreEstacion": "X",
                                    "Municipio": "Medellín"}, "geometry": {"coordinates": [-75.5, 6.2]}}]})
    assert geo.iloc[0]["municipio"] == "Medellín" and geo.iloc[0]["longitud"] == -75.5
    s = parse_series_72h({"info": {"Hora_fin": "2024-01-03 23:00:00", "Horas": [22, 23],
                                   "PM25_72H": [10, None]}}, 7)
    assert s["fecha"].tolist() == ["2024-01-03 22:00:00", "2024-01-03 23:00:00"]
    assert s["pm25"].iloc[1] == config.SIATA_MISSING


def _daily(values, start="2024-01-01", skip=()):
    dates = [d for i, d in enumerate(pd.date_range(start, periods=len(values))) if i not in skip]
    vals = [v for i, v in enumerate(values) if i not in skip]
    return pd.DataFrame({"codigo": 1, "fecha": dates, "pm25": vals,
                         "clase": [config.pm25_to_class(v) for v in vals]})


def test_features_target_is_next_day_without_leakage():
    df = build_features(_daily([10, 20, 30, 40, 60, 15]), ["log_pm25_t", "log_pm25_t1", "log_pm25_t2"])
    assert df["fecha"].dt.day.tolist() == [3, 4, 5]       # requiere t-2 y t+1
    assert df["y"].tolist() == [1, 2, 0]                  # clases de 40, 60, 15
    np.testing.assert_allclose(df["log_pm25_t"], np.log1p([30, 40, 60]))


def test_features_respect_calendar_gaps():
    # falta el día índice 2: los rezagos NO deben saltarse el hueco
    df = build_features(_daily([10, 20, 30, 40, 50, 60, 70], skip=(2,)),
                        ["log_pm25_t", "log_pm25_t1", "log_pm25_t2"])
    assert df["fecha"].dt.day.tolist() == [6]


def test_fast_f1_equals_sklearn():
    rng = np.random.default_rng(3)
    for _ in range(20):
        y, p = rng.integers(0, 3, 50), rng.integers(0, 3, 50)
        labels = sorted(set(y))
        assert np.isclose(fast_f1_macro(y, p),
                          f1_score(y, p, labels=labels, average="macro", zero_division=0))


def test_mcnemar_and_folds():
    y = np.zeros(30, int)
    r = mcnemar_exact(y, np.zeros(30, int), np.r_[np.ones(10, int), np.zeros(20, int)])
    assert r["b"] == 10 and r["c"] == 0 and r["p_value"] < 0.01
    dates = pd.date_range("2024-01-01", periods=100).values
    for tr, te in blocked_time_folds(dates, 5, 3):
        dtr, dte = dates[tr].astype("datetime64[D]"), dates[te].astype("datetime64[D]")
        gap = np.min(np.abs(dtr[:, None] - dte[None, :])).astype(int)
        assert gap > 3 and len(set(tr) & set(te)) == 0
