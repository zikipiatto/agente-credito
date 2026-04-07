"""
Scoring en producción — Modelo XGBoost Créditos Nuevos

Uso:
    from modelo_nuevos.predecir import predecir_decil
    resultado = predecir_decil(solicitud_dict)
    # resultado = {"decil": 3, "nivel_riesgo": "ALTO", "probabilidad_raw": 0.61}

El decil es RELATIVO a la distribución de entrenamiento (OOF).
Decil 1 = mayor riesgo | Decil 10 = menor riesgo.
Se usa el decil (no la probabilidad) para evitar el efecto del shift de mora entre periodos.
"""
import json
import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

# ── Rutas de artefactos ───────────────────────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELO_PATH   = os.path.join(_DIR, 'modelo.json')
_SCORING_PATH  = os.path.join(_DIR, 'modelo_scoring.json')

# ── Carga lazy de artefactos (una sola vez) ───────────────────────────
_model    = None
_scoring  = None


def _cargar():
    global _model, _scoring
    if _model is not None:
        return
    _model = XGBClassifier()
    _model.load_model(_MODELO_PATH)
    with open(_SCORING_PATH) as f:
        _scoring = json.load(f)


# ── Utilidades ────────────────────────────────────────────────────────
_NIVEL_RIESGO = {
    1: "MUY ALTO",
    2: "MUY ALTO",
    3: "ALTO",
    4: "ALTO",
    5: "MEDIO",
    6: "MEDIO",
    7: "BAJO",
    8: "BAJO",
    9: "MUY BAJO",
    10: "MUY BAJO",
}


def _calibrar(proba: float, cal_x: list, cal_y: list) -> float:
    """
    Aplica la calibración isotónica usando interpolación lineal sobre los
    puntos (cal_x, cal_y) guardados durante el entrenamiento.
    La calibradora fue entrenada sobre el hold-out, por lo que la probabilidad
    calibrada queda alineada con la tasa de mora real observada en ese periodo.
    """
    return float(np.interp(proba, cal_x, cal_y))


def _prob_a_decil(proba_cal: float, breakpoints: list) -> int:
    """
    Convierte una probabilidad CALIBRADA a decil.
    Breakpoints son 9 percentiles de las probs calibradas del hold-out (p10-p90).
    Probabilidades ALTAS = mayor riesgo = decil 1.
    """
    bp = sorted(breakpoints, reverse=True)
    for i, corte in enumerate(bp):
        if proba_cal >= corte:
            return i + 1
    return 10


def _preparar_fila(caso: dict, features: list, cat_features: list,
                   encoders: dict, sentinel: int) -> pd.DataFrame:
    """Convierte un dict de solicitud en un DataFrame de una fila listo para el modelo."""
    fila = {}
    for feat in features:
        fila[feat] = caso.get(feat, np.nan)

    X = pd.DataFrame([fila])

    # Categóricas
    for col in cat_features:
        val = str(X[col].iloc[0]) if pd.notna(X[col].iloc[0]) else 'SIN_DATO'
        classes = encoders.get(col, [])
        X[col] = classes.index(val) if val in classes else -1

    # Sentinels → NaN
    for col in features:
        if col not in cat_features:
            X[col] = pd.to_numeric(X[col], errors='coerce')
            if col in X.columns and X[col].iloc[0] == sentinel:
                X.loc[0, col] = np.nan

    X = X.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    return X


# ── Función principal ─────────────────────────────────────────────────
def predecir_decil(caso: dict) -> dict:
    """
    Recibe un dict con los campos de la solicitud y devuelve el decil de riesgo.

    Campos esperados (los que no estén se tratan como NaN):
        bc_score, icc, total_ingresos, total_egresos, creditos_vencidos,
        creditos_cerrados, creditos_activos, saldo_actual, monto_solicitado,
        tipo_vivienda, estado_civil

    Returns:
        {
            "decil":           int (1-10, donde 1=mayor riesgo, 10=menor riesgo),
            "nivel_riesgo":    str ("MUY ALTO" / "ALTO" / "MEDIO" / "BAJO" / "MUY BAJO"),
            "probabilidad_raw": float (probabilidad cruda del modelo, solo referencial),
        }
    """
    _cargar()

    features     = _scoring['features']
    cat_features = _scoring['cat_features']
    encoders     = _scoring['encoders']
    breakpoints  = _scoring['breakpoints']
    sentinel     = _scoring['sentinel']

    cal_x       = _scoring['calibradora_x']
    cal_y       = _scoring['calibradora_y']

    X = _preparar_fila(caso, features, cat_features, encoders, sentinel)
    proba_raw = float(_model.predict_proba(X)[0, 1])
    proba_cal = _calibrar(proba_raw, cal_x, cal_y)
    decil     = _prob_a_decil(proba_cal, breakpoints)

    return {
        "decil":                decil,
        "nivel_riesgo":         _NIVEL_RIESGO[decil],
        "probabilidad":         round(proba_cal, 4),   # calibrada — refleja mora real
        "probabilidad_raw":     round(proba_raw, 4),   # cruda — solo para monitoreo
    }


def predecir_desde_solicitud(solicitud: dict) -> dict:
    """
    Wrapper que extrae los campos relevantes de una SolicitudCompleta (dict)
    y llama a predecir_decil.

    El campo buro.score acepta tres formatos:
      - "588/0004"  → BC Score + ICC combinados (tipo BC SCORE / ICC en el reporte)
      - "588"       → solo BC Score
      - número      → score directo
    Si buro.icc también está presente, tiene precedencia sobre el ICC extraído del score.
    """
    buro    = solicitud.get('buro', {}) or {}
    fin     = solicitud.get('finanzas', {}) or {}
    cliente = solicitud.get('cliente', {}) or {}
    cond    = solicitud.get('condiciones', {}) or {}

    # Parsear bc_score e icc desde el campo score (puede venir combinado "588/0004")
    score_raw = buro.get('score')
    bc_score, icc_from_score = _parse_score_combinado(score_raw)

    # icc: prioridad al campo explícito buro.icc; si no, el extraído del score
    icc_explicito = buro.get('icc')
    icc = _parse_icc(icc_explicito) if icc_explicito is not None else icc_from_score

    caso = {
        'bc_score':          bc_score,
        'icc':               icc,
        'total_ingresos':    fin.get('total_ingresos'),
        'total_egresos':     fin.get('total_egresos'),
        'creditos_vencidos': buro.get('creditos_vencidos'),
        'creditos_cerrados': buro.get('creditos_cerrados'),
        'creditos_activos':  buro.get('creditos_activos'),
        'saldo_actual':      buro.get('saldo_actual'),
        'monto_solicitado':  cond.get('monto'),
        'tipo_vivienda':     cliente.get('tipo_vivienda'),
        'estado_civil':      cliente.get('estado_civil'),
    }

    return predecir_decil(caso)


def _parse_score_combinado(score_raw) -> tuple:
    """
    Parsea el campo score del buró que puede venir como:
      - "588/0004"  → (bc_score=588.0, icc=4.0)   — formato BC SCORE / ICC
      - "588"       → (bc_score=588.0, icc=NaN)
      - 588         → (bc_score=588.0, icc=NaN)
      - None / NaN  → (NaN, NaN)
    Devuelve (bc_score: float, icc: float).
    """
    if score_raw is None:
        return np.nan, np.nan
    try:
        s = str(score_raw).strip()
        if '/' in s:
            partes = s.split('/', 1)
            bc  = float(partes[0].strip())
            icc = float(partes[1].strip())
            return bc, icc
        else:
            return float(s), np.nan
    except (ValueError, TypeError):
        return np.nan, np.nan


def _parse_icc(icc_raw) -> float:
    """
    Parsea el ICC que puede venir como:
      - número o string simple: '4', 4, '0004'
      - formato combinado 'SCORE/ICC': '588/0004' → devuelve 4
    """
    if icc_raw is None:
        return np.nan
    try:
        s = str(icc_raw).strip()
        if '/' in s:
            s = s.split('/', 1)[1].strip()
        return float(s)
    except (ValueError, TypeError):
        return np.nan
