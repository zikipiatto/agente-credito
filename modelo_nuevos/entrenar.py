"""
Modelo de riesgo crediticio — CREDITOS NUEVOS
XGBoost + Optuna (tuning) + PSI + SHAP + IV + Calibración

Población:
  - Solo CREDITOS NUEVOS aprobados (con MaxAtr_ventana)
  - Solo ventana de observación cerrada: fecha_solicitud + 6 meses <= fecha corte
  - Hold-out fuera de tiempo: últimos MESES_HOLDOUT meses

Uso:
    python entrenar.py --input ruta/base.csv --output modelo.json [--trials 50]

Métricas objetivo:
    AUC > 0.75 | Gini > 0.50 | KS > 0.45 | Gap < 0.10 | PSI score < 0.10
"""
import argparse
import json
import warnings
import numpy as np
import pandas as pd
import optuna
import shap
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ─────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────
FEATURES = [
    # Score y comportamiento Buró (SHAP > 0.05)
    'bc_score', 'icc',
    # Financiero
    'total_ingresos', 'total_egresos',
    # Buró detalle
    'creditos_vencidos', 'creditos_cerrados', 'creditos_activos', 'saldo_actual',
    # Condiciones del crédito
    'monto_solicitado',
    # Perfil del cliente
    'tipo_vivienda', 'estado_civil',
]
CAT_FEATURES  = ['tipo_vivienda', 'estado_civil']
TARGET_COL    = 'MaxAtr_ventana'
TARGET_DIAS   = 30
SENTINEL      = -100
VENTANA_MESES = 6
MESES_HOLDOUT = 2
N_FOLDS       = 5


# ─────────────────────────────────────────
# Carga y filtrado
# ─────────────────────────────────────────
def cargar_datos(path: str):
    df = pd.read_csv(path, low_memory=False)
    df['fecha_solicitud'] = pd.to_datetime(df['fecha_solicitud'], dayfirst=True, errors='coerce')
    df = df[df['tipo_credito'] == 'CREDITOS NUEVOS'].copy()

    fecha_corte = df['fecha_solicitud'].max()
    print(f"Fecha de corte           : {fecha_corte.date()}")

    df['ventana_cerrada'] = df['fecha_solicitud'] + pd.DateOffset(months=VENTANA_MESES) <= fecha_corte

    total        = len(df)
    sin_ventana  = (~df['ventana_cerrada']).sum()
    rechazados   = (df['ventana_cerrada'] & df[TARGET_COL].isna()).sum()
    aprobados    = (df['ventana_cerrada'] & df[TARGET_COL].notna()).sum()

    print(f"Total créditos nuevos    : {total}")
    print(f"  Sin ventana cerrada    : {sin_ventana}  (excluidos)")
    print(f"  Rechazados sin comportamiento: {rechazados}  (excluidos)")
    print(f"  Aprobados con ventana  : {aprobados}  (usables)")

    df = df[df['ventana_cerrada'] & df[TARGET_COL].notna()].copy()
    df['target']   = (df[TARGET_COL] > TARGET_DIAS).astype(int)
    df['anio_mes'] = df['fecha_solicitud'].dt.to_period('M')

    print(f"\nMuestra final            : {len(df)}")
    print(f"  Buenos                 : {(df['target']==0).sum()}")
    print(f"  Malos                  : {(df['target']==1).sum()}")
    print(f"  Tasa mora DPD{TARGET_DIAS}+  : {df['target'].mean()*100:.1f}%")
    return df, fecha_corte


def separar_holdout(df: pd.DataFrame):
    meses = sorted(df['anio_mes'].unique())
    meses_ho    = meses[-MESES_HOLDOUT:]
    meses_train = meses[:-MESES_HOLDOUT]

    df_train   = df[df['anio_mes'].isin(meses_train)].copy()
    df_holdout = df[df['anio_mes'].isin(meses_ho)].copy()

    print(f"\nHold-out fuera de tiempo : {meses_ho[0]} → {meses_ho[-1]}")
    print(f"  Hold-out : {len(df_holdout):>6} casos | mora {df_holdout['target'].mean()*100:.1f}%")
    print(f"  Train    : {len(df_train):>6} casos | mora {df_train['target'].mean()*100:.1f}%")
    return df_train, df_holdout


# ─────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────
def construir_bc_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye bc_score e icc a partir de tipo_score + valor_score.

    Casos:
      tipo_score == "BC SCORE / ICC"  → valor_score = "588/0004" → bc_score=588, icc=4
      tipo_score == "BC SCORE"        → valor_score = "588"      → bc_score=588
      tipo_score == "SCORE NO HIT"    → usar score_no_hit si existe; bc_score queda como NaN
    """
    df = df.copy()
    tipo = df['tipo_score'].astype(str).str.strip().str.upper()
    valor = df['valor_score'].astype(str).str.strip()

    bc_list  = []
    icc_list = []

    tipo_num = pd.to_numeric(df['tipo_score'], errors='coerce')  # fallback: score en tipo_score

    for i, (t, v) in enumerate(zip(tipo, valor)):
        if 'ICC' in t and '/' in v:
            # Formato combinado "588/0004" → bc_score / ICC
            partes = v.split('/', 1)
            bc_list.append(pd.to_numeric(partes[0].strip(), errors='coerce'))
            icc_list.append(pd.to_numeric(partes[1].strip(), errors='coerce'))
        elif 'NO HIT' in t or 'SIN HIT' in t:
            # Sin historial: usar score_no_hit si existe; ICC no aplica
            bc_list.append(np.nan)
            icc_list.append(np.nan)
        else:
            bc = pd.to_numeric(v, errors='coerce')
            # Fallback: algunas bases tienen el score en tipo_score y valor_score vacío
            if pd.isna(bc):
                bc = tipo_num.iloc[i]
            bc_list.append(bc)
            icc_list.append(np.nan)

    df['bc_score'] = bc_list

    # Rellenar con score_no_hit donde bc_score quedó NaN (casos sin historial)
    if 'score_no_hit' in df.columns:
        sin_score = df['bc_score'].isna()
        df.loc[sin_score, 'bc_score'] = pd.to_numeric(
            df.loc[sin_score, 'score_no_hit'], errors='coerce'
        )

    # Rellenar icc con el extraído del score SOLO donde el campo icc está vacío
    icc_extraido = pd.Series(icc_list, index=df.index)
    tiene_icc_extraido = icc_extraido.notna()
    if 'icc' not in df.columns:
        df['icc'] = np.nan
    df['icc'] = pd.to_numeric(df['icc'], errors='coerce')
    # Solo rellenar donde icc es NaN (no pisar valores explícitos)
    sin_icc = df['icc'].isna()
    df.loc[tiene_icc_extraido & sin_icc, 'icc'] = icc_extraido[tiene_icc_extraido & sin_icc]

    n_con_icc = tiene_icc_extraido.sum()
    print(f"\nbc_score construido      : {df['bc_score'].notna().sum()} casos con score")
    print(f"icc extraído de valor_score: {n_con_icc} casos")
    print(f"icc total no-nulo        : {df['icc'].notna().sum()} casos")
    return df


def limpiar_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    cols = ['saldo_actual', 'saldo_vencido', 'creditos_activos', 'ingreso_neto', 'peor_atraso_dias']
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df.loc[df[col] == SENTINEL, col] = np.nan
    return df


def preparar_features(df: pd.DataFrame, encoders: dict = None, fit: bool = True) -> pd.DataFrame:
    X = df[FEATURES].copy()
    if encoders is None:
        encoders = {}
    for col in CAT_FEATURES:
        X[col] = X[col].fillna('SIN_DATO').astype(str)
        if fit:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            encoders[col] = le
        else:
            le = encoders[col]
            X[col] = X[col].map(lambda v: le.transform([v])[0] if v in le.classes_ else -1)
    if 'juicios_buro' in X.columns:
        X['juicios_buro'] = X['juicios_buro'].astype(int)
    X = X.apply(pd.to_numeric, errors='coerce')
    # Reemplazar inf/-inf por NaN (XGBoost no los acepta sin missing=inf)
    X = X.replace([np.inf, -np.inf], np.nan)
    return X, encoders


# ─────────────────────────────────────────
# Métricas
# ─────────────────────────────────────────
def ks_stat(y_true, y_proba) -> float:
    df_ks = pd.DataFrame({'t': y_true, 'p': y_proba}).sort_values('p', ascending=False)
    cb = (df_ks['t'] == 0).cumsum() / (y_true == 0).sum()
    cm = (df_ks['t'] == 1).cumsum() / (y_true == 1).sum()
    return float((cm.values - cb.values).max())


def psi(esperado: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """PSI entre dos distribuciones de score. esperado=train OOF, actual=holdout."""
    breakpoints = np.percentile(esperado, np.linspace(0, 100, bins + 1))
    breakpoints[0]  -= 1e-6
    breakpoints[-1] += 1e-6

    def bucket_pct(arr):
        counts, _ = np.histogram(arr, bins=breakpoints)
        pct = counts / len(arr)
        pct = np.where(pct == 0, 1e-6, pct)
        return pct

    e = bucket_pct(esperado)
    a = bucket_pct(actual)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_feature(train_vals: pd.Series, holdout_vals: pd.Series, bins: int = 10) -> float:
    """PSI de una feature continua entre train y holdout."""
    clean_tr = train_vals.dropna().values
    clean_ho = holdout_vals.dropna().values
    if len(clean_tr) == 0 or len(clean_ho) == 0:
        return np.nan
    return psi(clean_tr, clean_ho, bins)


def iv_feature(x: pd.Series, y: pd.Series, bins: int = 10) -> float:
    """Information Value de una feature respecto al target."""
    df = pd.DataFrame({'x': x, 'y': y}).dropna()
    if df['x'].nunique() <= 1:
        return 0.0
    try:
        df['bucket'] = pd.qcut(df['x'], bins, duplicates='drop')
    except Exception:
        df['bucket'] = df['x']
    tot_b = (df['y'] == 0).sum()
    tot_m = (df['y'] == 1).sum()
    if tot_b == 0 or tot_m == 0:
        return 0.0
    iv = 0.0
    for _, grp in df.groupby('bucket'):
        b = (grp['y'] == 0).sum() / tot_b
        m = (grp['y'] == 1).sum() / tot_m
        b = max(b, 1e-6); m = max(m, 1e-6)
        iv += (b - m) * np.log(b / m)
    return round(iv, 4)


def tabla_deciles(y_true, y_proba, titulo="TABLA DE DECILES"):
    res = pd.DataFrame({'target': y_true, 'proba': y_proba})
    res = res.sort_values('proba', ascending=False).reset_index(drop=True)
    res['decil'] = pd.qcut(res['proba'], 10, labels=False, duplicates='drop')
    res['decil'] = 9 - res['decil']
    total_m = res['target'].sum()
    print(f"\n=== {titulo} ===")
    print(f"  {'Decil':>6} {'N':>6} {'Malos':>7} {'Mora%':>7} {'% Malos acum':>14}")
    print("  " + "-" * 46)
    acum = 0
    for d in range(10):
        sub = res[res['decil'] == d]
        m   = sub['target'].sum()
        acum += m
        mora = m / len(sub) * 100 if len(sub) > 0 else 0
        print(f"  {d+1:>6} {len(sub):>6} {m:>7} {mora:>6.1f}% {acum/total_m*100:>13.1f}%")


# ─────────────────────────────────────────
# Optuna — tuning de hiperparámetros
# ─────────────────────────────────────────
def tune_hiperparametros(X: pd.DataFrame, y: pd.Series, n_trials: int = 50) -> dict:
    scale = (y == 0).sum() / (y == 1).sum()
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            'max_depth':        trial.suggest_int('max_depth', 2, 6),
            'n_estimators':     trial.suggest_int('n_estimators', 100, 500),
            'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'subsample':        trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 5, 60),
            'reg_lambda':       trial.suggest_float('reg_lambda', 1, 20),
            'reg_alpha':        trial.suggest_float('reg_alpha', 0, 5),
            'scale_pos_weight': scale,
            'random_state': 42, 'verbosity': 0,
        }
        model = XGBClassifier(**params)
        aucs  = []
        for tr_idx, val_idx in cv.split(X, y):
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            p = model.predict_proba(X.iloc[val_idx])[:, 1]
            aucs.append(roc_auc_score(y.iloc[val_idx], p))
        return np.mean(aucs)

    print(f"\n=== OPTUNA TUNING ({n_trials} trials) ===")
    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    best['scale_pos_weight'] = scale
    best['random_state']     = 42
    best['verbosity']        = 0

    print(f"  Mejor AUC CV  : {study.best_value:.4f}")
    print(f"  Mejores params: {json.dumps({k:v for k,v in best.items() if k not in ['scale_pos_weight','random_state','verbosity']}, indent=4)}")
    return best


# ─────────────────────────────────────────
# Entrenamiento final con CV
# ─────────────────────────────────────────
def entrenar_cv(X: pd.DataFrame, y: pd.Series, params: dict):
    model = XGBClassifier(**params)
    cv    = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    tr_aucs, val_aucs, ks_list, oof_proba = [], [], [], np.zeros(len(y))

    print(f"\n=== VALIDACIÓN CRUZADA {N_FOLDS}-FOLD (params óptimos) ===")
    print(f"  {'Fold':<6} {'AUC train':>10} {'AUC val':>10} {'Gini':>8} {'KS':>8}")
    print("  " + "-" * 46)

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y)):
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        p_tr  = model.predict_proba(X.iloc[tr_idx])[:, 1]
        p_val = model.predict_proba(X.iloc[val_idx])[:, 1]
        oof_proba[val_idx] = p_val
        auc_tr  = roc_auc_score(y.iloc[tr_idx], p_tr)
        auc_val = roc_auc_score(y.iloc[val_idx], p_val)
        ks_val  = ks_stat(y.iloc[val_idx].values, p_val)
        tr_aucs.append(auc_tr); val_aucs.append(auc_val); ks_list.append(ks_val)
        print(f"  {fold+1:<6} {auc_tr:>10.3f} {auc_val:>10.3f} {2*auc_val-1:>8.3f} {ks_val:>8.3f}")

    gap = np.mean(tr_aucs) - np.mean(val_aucs)
    print("  " + "-" * 46)
    print(f"  {'Prom':<6} {np.mean(tr_aucs):>10.3f} {np.mean(val_aucs):>10.3f} {2*np.mean(val_aucs)-1:>8.3f} {np.mean(ks_list):>8.3f}")
    print(f"  {'Std':<6} {'':>10} {np.std(val_aucs):>10.3f} {2*np.std(val_aucs):>8.3f} {np.std(ks_list):>8.3f}")
    print(f"\n  Gap train-val  : {gap:.3f} ", end="")
    print("✅" if gap < 0.05 else ("⚠️  leve" if gap < 0.10 else "❌ sobreajuste"))

    tabla_deciles(y.values, oof_proba, f"DECILES — VALIDACIÓN CRUZADA (OOF)")

    # Entrenar modelo final con todos los datos de train
    model.fit(X, y)

    metricas = {
        "auc_cv":    round(float(np.mean(val_aucs)), 4),
        "gini_cv":   round(float(2 * np.mean(val_aucs) - 1), 4),
        "ks_cv":     round(float(np.mean(ks_list)), 4),
        "auc_std":   round(float(np.std(val_aucs)), 4),
        "gap":       round(float(gap), 4),
        "n_train":   int(len(y)),
        "n_malos":   int(y.sum()),
        "tasa_mora": round(float(y.mean() * 100), 2),
    }
    return model, metricas, oof_proba


# ─────────────────────────────────────────
# Calibración de probabilidades
# ─────────────────────────────────────────
def calibrar_modelo(model, X: pd.DataFrame, y: pd.Series):
    """Calibración isotónica — entrena un calibrador sobre las predicciones del modelo."""
    from sklearn.isotonic import IsotonicRegression
    p_raw = model.predict_proba(X)[:, 1]
    calibrador = IsotonicRegression(out_of_bounds='clip')
    calibrador.fit(p_raw, y)
    p_cal = calibrador.predict(p_raw)

    brier_raw = brier_score_loss(y, p_raw)
    brier_cal = brier_score_loss(y, p_cal)

    print(f"\n=== CALIBRACIÓN (Brier Score — menor es mejor) ===")
    print(f"  Sin calibrar : {brier_raw:.4f}")
    print(f"  Calibrado    : {brier_cal:.4f}")
    return calibrador


# ─────────────────────────────────────────
# SHAP
# ─────────────────────────────────────────
def analizar_shap(model, X: pd.DataFrame, n: int = 500):
    print(f"\n=== SHAP — TOP 15 FEATURES (muestra {n} casos) ===")
    muestra = X.sample(min(n, len(X)), random_state=42)
    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(muestra)
    importancia = pd.Series(
        np.abs(shap_vals).mean(axis=0),
        index=X.columns
    ).sort_values(ascending=False)
    print(f"  {'Feature':<35} {'SHAP medio abs':>15}")
    print("  " + "-" * 52)
    for feat, val in importancia.head(15).items():
        print(f"  {feat:<35} {val:>15.4f}")
    return importancia


# ─────────────────────────────────────────
# PSI y IV
# ─────────────────────────────────────────
def reporte_psi_iv(X_train: pd.DataFrame, y_train: pd.Series,
                   X_holdout: pd.DataFrame, y_holdout: pd.Series,
                   oof_proba: np.ndarray, p_holdout: np.ndarray):

    # PSI del score
    psi_score = psi(oof_proba, p_holdout)
    estado_psi = "✅ estable" if psi_score < 0.10 else ("⚠️  moderado" if psi_score < 0.25 else "❌ inestable")
    print(f"\n=== PSI DEL SCORE (train OOF vs holdout) ===")
    print(f"  PSI = {psi_score:.4f}  {estado_psi}")
    print(f"  Referencia: <0.10 estable | 0.10-0.25 revisar | >0.25 reentrenar")

    # PSI + IV por feature
    print(f"\n=== PSI Y IV POR FEATURE ===")
    print(f"  {'Feature':<35} {'PSI':>8} {'IV':>8}  {'Estado PSI'}")
    print("  " + "-" * 65)
    resultados = []
    for col in X_train.columns:
        p = psi_feature(X_train[col], X_holdout[col])
        v = iv_feature(X_train[col], y_train)
        est = "✅" if (np.isnan(p) or p < 0.10) else ("⚠️" if p < 0.25 else "❌")
        resultados.append({'feature': col, 'psi': p, 'iv': v, 'estado': est})

    resultados = sorted(resultados, key=lambda x: x['iv'] if not np.isnan(x['iv']) else 0, reverse=True)
    for r in resultados:
        psi_str = f"{r['psi']:.4f}" if not np.isnan(r['psi']) else "  N/A "
        print(f"  {r['feature']:<35} {psi_str:>8} {r['iv']:>8.4f}  {r['estado']}")

    return psi_score, resultados


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",   default="/Users/fernandosierra/proyectos/agente-credito/base_motor_decision_crediticia_2024_2025 1.csv")
    parser.add_argument("--output",  default="modelo_nuevos/modelo.json")
    parser.add_argument("--trials",  type=int, default=50, help="Número de trials Optuna")
    parser.add_argument("--exclude", default=None, help="Archivo .txt o CSV con numero_solicitud a excluir (uno por línea)")
    args = parser.parse_args()

    # ── Carga y preparación ──
    df, fecha_corte = cargar_datos(args.input)

    if args.exclude:
        with open(args.exclude) as fex:
            excluir = set(int(x.strip()) for x in fex if x.strip().isdigit())
        antes = len(df)
        df = df[~df['numero_solicitud'].isin(excluir)].copy()
        print(f"\nExcluidos {antes - len(df)} casos de muestra de prueba ({args.exclude})")
    df = construir_bc_score(df)
    df = limpiar_sentinels(df)
    df_train, df_holdout = separar_holdout(df)

    encoders = {}
    X_train, encoders = preparar_features(df_train, encoders=encoders, fit=True)
    y_train = df_train['target'].reset_index(drop=True)
    X_train = X_train.reset_index(drop=True)

    X_holdout, _ = preparar_features(df_holdout, encoders=encoders, fit=False)
    y_holdout = df_holdout['target'].reset_index(drop=True)
    X_holdout = X_holdout.reset_index(drop=True)

    nas = X_train.isna().sum()
    nas = nas[nas > 0]
    if len(nas):
        print(f"\nNAs en entrenamiento:\n{nas.to_string()}")

    # ── Tuning ──
    best_params = tune_hiperparametros(X_train, y_train, n_trials=args.trials)

    # ── Entrenamiento con CV ──
    model, metricas, oof_proba = entrenar_cv(X_train, y_train, best_params)

    # ── Hold-out ──
    p_holdout = model.predict_proba(X_holdout)[:, 1]
    auc_ho  = roc_auc_score(y_holdout, p_holdout)
    ks_ho   = ks_stat(y_holdout.values, p_holdout)
    gini_ho = 2 * auc_ho - 1

    print(f"\n=== HOLD-OUT FUERA DE TIEMPO ===")
    print(f"  AUC={auc_ho:.3f} | Gini={gini_ho:.3f} | KS={ks_ho:.3f}")
    tabla_deciles(y_holdout.values, p_holdout, "DECILES — HOLD-OUT")

    metricas.update({
        "auc_holdout":  round(auc_ho, 4),
        "gini_holdout": round(gini_ho, 4),
        "ks_holdout":   round(ks_ho, 4),
        "n_holdout":    int(len(y_holdout)),
    })

    # ── Calibración ──
    modelo_calibrado = calibrar_modelo(model, X_train, y_train)

    # ── SHAP ──
    analizar_shap(model, X_train)

    # ── PSI + IV ──
    psi_score, reporte_features = reporte_psi_iv(
        X_train, y_train, X_holdout, y_holdout, oof_proba, p_holdout
    )
    metricas['psi_score'] = round(psi_score, 4)

    # ── Calibración isotónica sobre el hold-out (mora real observada) ──
    # Entrenamos la calibradora con las probabilidades del hold-out y sus targets reales.
    # Así la probabilidad calibrada queda alineada con la tasa de mora del periodo actual,
    # no con la tasa del entrenamiento (que puede diferir por drift temporal).
    from sklearn.isotonic import IsotonicRegression
    calibradora = IsotonicRegression(out_of_bounds='clip')
    calibradora.fit(p_holdout, y_holdout)

    p_holdout_cal = calibradora.predict(p_holdout)
    brier_raw = float(np.mean((p_holdout - y_holdout.values) ** 2))
    brier_cal = float(np.mean((p_holdout_cal - y_holdout.values) ** 2))
    print(f"\n=== CALIBRACIÓN ISOTÓNICA (hold-out como referencia) ===")
    print(f"  Mora real hold-out : {y_holdout.mean()*100:.1f}%")
    print(f"  Prob media s/cal   : {p_holdout.mean()*100:.1f}%")
    print(f"  Prob media c/cal   : {p_holdout_cal.mean()*100:.1f}%")
    print(f"  Brier sin calibrar : {brier_raw:.4f}")
    print(f"  Brier calibrado    : {brier_cal:.4f}")

    # Guardar calibradora como tabla de interpolación (sin pickle, solo JSON)
    cal_x = calibradora.X_thresholds_.tolist()
    cal_y = calibradora.y_thresholds_.tolist()

    # ── Cortes de deciles sobre probabilidades CALIBRADAS del hold-out ──
    # Decil 1 = mayor riesgo, Decil 10 = menor riesgo
    breakpoints = np.percentile(p_holdout_cal, np.linspace(10, 90, 9)).tolist()

    # ── Encoders de variables categóricas ──
    encoders_export = {col: le.classes_.tolist() for col, le in encoders.items()}

    # ── Artefactos de scoring ──
    scoring_artifacts = {
        "features":       FEATURES,
        "cat_features":   CAT_FEATURES,
        "sentinel":       SENTINEL,
        "calibradora_x":  cal_x,   # breakpoints X de la isotónica
        "calibradora_y":  cal_y,   # valores Y de la isotónica
        "breakpoints":    breakpoints,   # 9 cortes para deciles 1-10 (sobre prob calibrada)
        "encoders":       encoders_export,
        "mora_holdout":   round(float(y_holdout.mean() * 100), 2),
        "mora_train":     round(float(y_train.mean() * 100), 2),
        "nota": "Probabilidad calibrada con isotónica sobre hold-out. Refleja mora real del periodo."
    }

    # ── Guardar ──
    model.save_model(args.output)

    metricas_path = args.output.replace('.json', '_metricas.json')
    with open(metricas_path, 'w') as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)

    holdout_path = args.output.replace('.json', '_holdout.csv')
    df_holdout.to_csv(holdout_path, index=False)

    params_path = args.output.replace('.json', '_params.json')
    with open(params_path, 'w') as f:
        json.dump({k: (float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v)
                   for k, v in best_params.items()}, f, indent=2)

    scoring_path = args.output.replace('.json', '_scoring.json')
    with open(scoring_path, 'w') as f:
        json.dump(scoring_artifacts, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*55}")
    print(f"✅ Modelo guardado        : {args.output}")
    print(f"✅ Métricas               : {metricas_path}")
    print(f"✅ Hiperparámetros        : {params_path}")
    print(f"✅ Hold-out               : {holdout_path}")
    print(f"✅ Artefactos scoring     : {scoring_path}")
    print(f"\nCortes de deciles (OOF): {[round(b,4) for b in breakpoints]}")
    print(f"\nResumen CV  : AUC={metricas['auc_cv']} ± {metricas['auc_std']} | Gini={metricas['gini_cv']} | KS={metricas['ks_cv']} | Gap={metricas['gap']}")
    print(f"Resumen HO  : AUC={metricas['auc_holdout']} | Gini={metricas['gini_holdout']} | KS={metricas['ks_holdout']}")
    print(f"PSI score   : {metricas['psi_score']}  {'✅ estable' if psi_score < 0.10 else '⚠️  revisar' if psi_score < 0.25 else '❌ inestable'}")


if __name__ == "__main__":
    main()
