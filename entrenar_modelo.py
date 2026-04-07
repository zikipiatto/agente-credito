"""
Limpieza de datos y reentrenamiento del modelo de decisión crediticia.
Base: base_motor_decision_crediticia_2024_2025 1.csv

Estrategia:
- Solo créditos efectivamente otorgados (tienen No_Credito y MaxAtr_ventana)
- Target binario: MaxAtr_ventana >= 30 días = 1 (malo)
- Limpieza profunda de centinelas (-100, -10), infinitos y constantes
- Modelo: LightGBM (maneja NaN nativo, mejor en datos tabulares desbalanceados)
- Validación: StratifiedKFold 5-fold + métricas de negocio (KS, Gini, recall en malos)
"""

import warnings, json, os
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (roc_auc_score, classification_report,
                             confusion_matrix, roc_curve)
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

# ── Rutas ─────────────────────────────────────────────────────────
BASE_DIR  = Path('/Users/fernandosierra/proyectos/agente-credito')
DATA_PATH = BASE_DIR / 'base_motor_decision_crediticia_2024_2025 1.csv'
OUT_DIR   = BASE_DIR / 'modelo'
OUT_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("  MOTOR DE DECISIÓN CREDITICIA — REENTRENAMIENTO")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════
# 1. CARGA
# ══════════════════════════════════════════════════════════════════
print("\n[1] Cargando datos...")
df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"    Raw: {df.shape[0]:,} registros x {df.shape[1]} columnas")

# ══════════════════════════════════════════════════════════════════
# 2. FILTRAR CRÉDITOS CON VENTANA DE OBSERVACIÓN COMPLETA
# ══════════════════════════════════════════════════════════════════
print("\n[2] Filtrando créditos con ventana de observación completa...")
df = df[df['No_Credito'].notna()].copy()
df = df[df['MaxAtr_ventana'].notna()].copy()
print(f"    Registros con target: {df.shape[0]:,}")

# Ventana de desempeño = 26 semanas = 182 días
VENTANA_DIAS = 182
df['fecha_dt'] = pd.to_datetime(df['fecha_solicitud'], dayfirst=True, errors='coerce')
fecha_max_data = df['fecha_dt'].max()
df['fecha_fin_ventana'] = df['fecha_dt'] + pd.Timedelta(days=VENTANA_DIAS)
df['ventana_completa'] = df['fecha_fin_ventana'] <= fecha_max_data

n_antes = len(df)
df = df[df['ventana_completa']].copy()
n_excluidos = n_antes - len(df)
print(f"    Fecha máxima en datos: {fecha_max_data.date()}")
print(f"    Corte para ventana completa: créditos hasta {(fecha_max_data - pd.Timedelta(days=VENTANA_DIAS)).date()}")
print(f"    Excluidos por ventana incompleta: {n_excluidos:,} ({n_excluidos/n_antes:.1%}) — ago-2025 en adelante")
print(f"    Dataset con ventana completa: {len(df):,}")

# ── Excluir productos que no son crédito de buró
# 15106 = Nómina (sin ICC, tasa malos 0.6% — perfil completamente distinto)
# 15109 = ToCuenta (otro producto, sin BC Score)
n_antes2 = len(df)
df = df[~df['producto'].isin([15106, 15109])].copy()
print(f"    Excluidos 15106 (Nómina) y 15109 (ToCuenta): {n_antes2 - len(df):,}")

# ── Solo créditos nuevos (excluir renovaciones y recompras)
n_antes3 = len(df)
df = df[df['tipo_credito'] == 'CREDITOS NUEVOS'].copy()
print(f"    Excluidos renovaciones y recompras: {n_antes3 - len(df):,}")
print(f"    Dataset limpio para entrenamiento: {len(df):,}")

# ── Target binario
df['malo'] = (df['MaxAtr_ventana'] >= 30).astype(int)
tasa_mala = df['malo'].mean()
print(f"    Tasa de malos (>=30d): {tasa_mala:.1%}  ({df['malo'].sum():,} malos / {len(df):,} total)")

# ══════════════════════════════════════════════════════════════════
# 3. LIMPIEZA PROFUNDA
# ══════════════════════════════════════════════════════════════════
print("\n[3] Limpieza de datos...")

# ── 3.1 Eliminar columnas inútiles
DROP_COLS = [
    # Identificadores / PII
    'numero_solicitud', 'No_Credito', 'nombre_prospecto', 'rfc', 'curp',
    'fecha_nacimiento', 'folio_buro', 'nombre_negocio', 'fecha_solicitud',
    # Constantes (todos = 100)
    'score_verificacion_id', 'score_reconocimiento_facial', 'score_deteccion_vida',
    'score_validacion_gobierno', 'score_video_selfie',
    # Cuasi-constante (99.99% = 0)
    'ingreso_neto',
    # 94.5% nulos
    'fico_score',
    # Target (no usar como feature)
    'MaxAtr_ventana', 'malo',
    # Columnas auxiliares de fecha (usadas solo para filtro de ventana)
    'fecha_dt', 'fecha_fin_ventana', 'ventana_completa',
    # ── LEAKAGE: salidas de modelos — NO usar como features ──────
    # Salida del modelo actual recalibrado
    'probabilidad', 'probabilidad_api', 'distribucion', 'decil_prod_bc',
    # Salidas de modelos anteriores
    'decil_ml', 'decision_ml',
    # Salida de modelo de regresión de capacidad de pago
    'capacidad_pago_ml',
    # cuota_ingreso_pct derivada de capacidad_pago_ml
    'cuota_ingreso_pct',
    # ── VARIABLES SIN APORTE (gain=0, IV≈0) ──────────────────────
    'genero',          # IV=0, gain<0.1%
    'frecuencia',      # IV=0, gain<0.1% — colineal con plazo y producto
    'distancia_km',    # IV=0, gain<0.2% — solo 3 valores artificiales
    'juicios_buro',    # IV=0, gain<0.05% — casi sin variación
    'tipo_score',      # gain=0 — codifica tipo de score pero enmascarado por valor_score
]
df_feat = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors='ignore')

# ── 3.2 Centinelas numéricos → NaN
SENTINEL_COLS = [
    'creditos_activos', 'saldo_actual', 'saldo_vencido',
    'valor_score', 'score_no_hit', 'vanohi',
]
for col in SENTINEL_COLS:
    if col in df_feat.columns:
        df_feat[col] = pd.to_numeric(df_feat[col], errors='coerce')
        df_feat[col] = df_feat[col].replace([-100, -10, -1], np.nan)

# ── ICC: tratamiento diferenciado
# ICC == -3 no significa "sin buró" — la mayoría tiene créditos cerrados.
# Significa que el sistema no asignó ICC (sin score de capacidad).
# Creamos flag binario y dejamos -3 como NaN en la escala ordinal.
if 'icc' in df_feat.columns:
    df_feat['icc'] = pd.to_numeric(df_feat['icc'], errors='coerce')
    df_feat['flag_sin_icc'] = (df_feat['icc'] == -3).astype(int)
    df_feat['icc'] = df_feat['icc'].replace(-3, np.nan)  # -3 no es un punto de la escala 0-9

# ── 3.3 peor_atraso_dias: texto con '-100' y espacios
if 'peor_atraso_dias' in df_feat.columns:
    df_feat['peor_atraso_dias'] = pd.to_numeric(
        df_feat['peor_atraso_dias'].astype(str).str.strip().replace('-100', np.nan),
        errors='coerce'
    )
    df_feat['peor_atraso_dias'] = df_feat['peor_atraso_dias'].replace(-100, np.nan)

# ── 3.4 cuota_ingreso_pct: infinitos
# ── 3.5 tipo_score: limpiar (contiene scores como strings '604', 'Sin Dato', '0')
if 'tipo_score' in df_feat.columns:
    def limpiar_tipo_score(v):
        v = str(v).strip()
        if v in ('Sin Dato', '0', 'nan', ''):
            return 'SIN_DATO'
        try:
            float(v)  # es un número → probablemente BC Score mal mapeado
            return 'BC_SCORE_NUM'
        except:
            return v.upper()
    df_feat['tipo_score'] = df_feat['tipo_score'].apply(limpiar_tipo_score)

# ── 3.6 decision_ml: limpiar categorías
if 'decision_ml' in df_feat.columns:
    df_feat['decision_ml'] = df_feat['decision_ml'].astype(str).str.strip()
    df_feat['decision_ml'] = df_feat['decision_ml'].replace({'Sin Dato': 'SIN_DATO', 'nan': 'SIN_DATO'})

# ── 3.7 decil_ml: 'Sin Dato' → NaN
# decil_ml y decision_ml eliminadas (leakage de modelos anteriores)

# ── 3.8 Columnas de texto → categorías numéricas
# Identificar columnas object restantes
cat_cols = df_feat.select_dtypes(include='object').columns.tolist()
print(f"    Columnas categóricas a codificar: {cat_cols}")

label_encoders = {}
for col in cat_cols:
    df_feat[col] = df_feat[col].astype(str).fillna('MISSING')
    le = LabelEncoder()
    df_feat[col] = le.fit_transform(df_feat[col])
    label_encoders[col] = le

# ── 3.9 Eliminar columnas con > 80% nulos después de limpieza
null_pct = df_feat.isnull().mean()
cols_drop_null = null_pct[null_pct > 0.80].index.tolist()
if cols_drop_null:
    print(f"    Eliminando por >80% nulos: {cols_drop_null}")
    df_feat = df_feat.drop(columns=cols_drop_null)

# ── 3.10 Eliminar columnas con varianza 0 o cuasi-cero
nunique = df_feat.nunique()
cols_const = nunique[nunique <= 1].index.tolist()
if cols_const:
    print(f"    Eliminando constantes: {cols_const}")
    df_feat = df_feat.drop(columns=cols_const)

print(f"    Features finales: {df_feat.shape[1]} columnas")

# ── 3.11 Distancia_km: solo 3 valores posibles, probablemente generada artificialmente
# La mantenemos pero con baja importancia esperada

# ══════════════════════════════════════════════════════════════════
# 4. INGENIERÍA DE FEATURES ADICIONALES
# ══════════════════════════════════════════════════════════════════
print("\n[4] Ingeniería de features...")

# Ratio saldo / créditos activos
if 'saldo_actual' in df_feat.columns and 'creditos_activos' in df_feat.columns:
    df_feat['ratio_saldo_creditos'] = (
        df_feat['saldo_actual'] / (df_feat['creditos_activos'] + 1)
    ).clip(0, 1e6)

# Capacidad real (ingreso - egreso)
if 'total_ingresos' in df_feat.columns and 'total_egresos' in df_feat.columns:
    df_feat['total_ingresos'] = pd.to_numeric(df_feat['total_ingresos'], errors='coerce')
    df_feat['total_egresos']  = pd.to_numeric(df_feat['total_egresos'],  errors='coerce')
    df_feat['capacidad_real'] = (df_feat['total_ingresos'] - df_feat['total_egresos']).clip(-1e6, 1e6)

# flag_sin_icc ya fue creado en el bloque de ICC arriba

print(f"    Features totales (con ingeniería): {df_feat.shape[1]}")

# ══════════════════════════════════════════════════════════════════
# 5. PREPARAR MATRICES X, y
# ══════════════════════════════════════════════════════════════════
y = df['malo'].values
X = df_feat.values
feature_names = df_feat.columns.tolist()

print(f"\n[5] Dataset final: X={X.shape}, positivos={y.sum():,} ({y.mean():.1%})")

# ══════════════════════════════════════════════════════════════════
# 6. ENTRENAMIENTO CON VALIDACIÓN CRUZADA
# ══════════════════════════════════════════════════════════════════
print("\n[6] Entrenamiento LightGBM con StratifiedKFold 5-fold...")

scale_pos = (y == 0).sum() / (y == 1).sum()

params = {
    'objective':       'binary',
    'metric':          'auc',
    'n_estimators':    800,
    'learning_rate':   0.05,
    'num_leaves':      63,
    'max_depth':       6,
    'min_child_samples': 50,
    'subsample':       0.8,
    'colsample_bytree': 0.8,
    'reg_alpha':       0.1,
    'reg_lambda':      1.0,
    'scale_pos_weight': scale_pos,
    'random_state':    42,
    'n_jobs':          -1,
    'verbose':         -1,
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_proba = np.zeros(len(y))
fold_aucs  = []

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    oof_proba[val_idx] = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, oof_proba[val_idx])
    fold_aucs.append(auc)
    print(f"    Fold {fold}: AUC={auc:.4f}")

print(f"\n    OOF AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")

# KS statistic
fpr, tpr, thresholds = roc_curve(y, oof_proba)
ks = np.max(tpr - fpr)
gini = 2 * roc_auc_score(y, oof_proba) - 1
print(f"    KS:   {ks:.4f}")
print(f"    Gini: {gini:.4f}")

# ══════════════════════════════════════════════════════════════════
# 7. MODELO FINAL — train en todos los datos
# ══════════════════════════════════════════════════════════════════
print("\n[7] Entrenando modelo final en todos los datos...")

X_tr_f, X_val_f, y_tr_f, y_val_f = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
model_final = lgb.LGBMClassifier(**params)
model_final.fit(
    X_tr_f, y_tr_f,
    eval_set=[(X_val_f, y_val_f)],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
)

test_proba = model_final.predict_proba(X_val_f)[:, 1]
test_auc   = roc_auc_score(y_val_f, test_proba)
print(f"    AUC test hold-out: {test_auc:.4f}")

# ══════════════════════════════════════════════════════════════════
# 8. ANÁLISIS POR DECIL (tabla de lift)
# ══════════════════════════════════════════════════════════════════
print("\n[8] Tabla de deciles (hold-out)...")

df_eval = pd.DataFrame({'prob': test_proba, 'malo': y_val_f})
df_eval['decil'] = pd.qcut(df_eval['prob'], q=10, labels=False, duplicates='drop') + 1
tabla = df_eval.groupby('decil').agg(
    n=('malo', 'count'),
    malos=('malo', 'sum'),
    tasa_mala=('malo', 'mean'),
    prob_media=('prob', 'mean'),
).sort_index(ascending=False)
tabla['tasa_mala_pct'] = (tabla['tasa_mala'] * 100).round(1)
print(tabla[['n', 'malos', 'tasa_mala_pct', 'prob_media']].to_string())

# ══════════════════════════════════════════════════════════════════
# 9. IMPORTANCIA DE FEATURES
# ══════════════════════════════════════════════════════════════════
print("\n[9] Top 20 features por importancia...")
importances = model_final.feature_importances_
feat_imp = pd.DataFrame({'feature': feature_names, 'importance': importances})
feat_imp = feat_imp.sort_values('importance', ascending=False).head(20)
print(feat_imp.to_string(index=False))

# ══════════════════════════════════════════════════════════════════
# 10. UMBRAL ÓPTIMO PARA EL MOTOR DE DECISIÓN
# ══════════════════════════════════════════════════════════════════
print("\n[10] Calibrando umbral de decisión...")

# Buscar umbral que capture >= 70% de malos (recall) con máxima precisión
from sklearn.metrics import precision_recall_curve
prec, rec, thr = precision_recall_curve(y_val_f, test_proba)
# precision_recall_curve devuelve len(prec) = len(rec) = len(thr) + 1
prec_t = prec[:-1]
rec_t  = rec[:-1]

# Umbral con recall >= 0.70 y máxima precisión
mask = rec_t >= 0.70
if mask.any():
    best_idx  = np.argmax(prec_t[mask])
    best_thr  = thr[mask][best_idx]
    best_prec = prec_t[mask][best_idx]
    best_rec  = rec_t[mask][best_idx]
else:
    best_thr = 0.30
    best_prec = best_rec = 0.0

print(f"    Umbral recomendado: {best_thr:.3f}")
print(f"    Precision en malos: {best_prec:.1%}")
print(f"    Recall en malos:    {best_rec:.1%}")

# Mapeo decil → acción del motor
DECIL_ACCION = {
    10: 'APROBADO',
    9:  'APROBADO',
    8:  'APROBADO',
    7:  'VALIDACION',
    6:  'VALIDACION',
    5:  'VALIDACION',
    4:  'ESCALAR_EJECUTIVO',
    3:  'ESCALAR_EJECUTIVO',
    2:  'RECHAZADO',
    1:  'RECHAZADO',
}

# Ajustar según tasa de mala observada por decil
print("\n    Mapeo decil → acción:")
for d in range(10, 0, -1):
    fila = tabla.loc[d] if d in tabla.index else None
    tasa = f"{fila['tasa_mala_pct']:.1f}%" if fila is not None else "?"
    print(f"    Decil {d:2d}: tasa_mala={tasa:>6}  →  {DECIL_ACCION[d]}")

# ══════════════════════════════════════════════════════════════════
# 11. GUARDAR ARTEFACTOS
# ══════════════════════════════════════════════════════════════════
print("\n[11] Guardando modelo y artefactos...")

joblib.dump(model_final,    OUT_DIR / 'lgbm_credito.pkl')
joblib.dump(label_encoders, OUT_DIR / 'label_encoders.pkl')
joblib.dump(feature_names,  OUT_DIR / 'feature_names.pkl')

meta = {
    'feature_names':   feature_names,
    'n_features':      len(feature_names),
    'oof_auc':         float(np.mean(fold_aucs)),
    'test_auc':        float(test_auc),
    'ks':              float(ks),
    'gini':            float(gini),
    'umbral_decision': float(best_thr),
    'precision_malos': float(best_prec),
    'recall_malos':    float(best_rec),
    'tasa_mala_train': float(tasa_mala),
    'n_train':         int(len(y)),
    'decil_accion':    DECIL_ACCION,
    'top_features':    feat_imp['feature'].head(10).tolist(),
    'params':          params,
}
with open(OUT_DIR / 'modelo_meta.json', 'w') as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

# ── Gráfica: ROC curve + importancia
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ROC
axes[0].plot(fpr, tpr, color='#1a73e8', lw=2, label=f'AUC OOF = {np.mean(fold_aucs):.3f}')
axes[0].plot([0,1],[0,1], 'k--', lw=1)
axes[0].set_xlabel('Tasa de Falsos Positivos')
axes[0].set_ylabel('Tasa de Verdaderos Positivos')
axes[0].set_title('Curva ROC — Modelo LightGBM')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Feature importance
feat_imp_plot = feat_imp.head(15).sort_values('importance')
axes[1].barh(feat_imp_plot['feature'], feat_imp_plot['importance'], color='#1a73e8')
axes[1].set_title('Top 15 Features por Importancia')
axes[1].set_xlabel('Importance (gain)')
axes[1].grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(OUT_DIR / 'modelo_evaluacion.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\n{'='*70}")
print(f"  LISTO. Artefactos guardados en: {OUT_DIR}/")
print(f"{'='*70}")
print(f"  lgbm_credito.pkl      — modelo serializado")
print(f"  label_encoders.pkl    — encoders de variables categóricas")
print(f"  feature_names.pkl     — lista de features en orden")
print(f"  modelo_meta.json      — métricas y configuración")
print(f"  modelo_evaluacion.png — curva ROC + importancia")
print(f"{'='*70}")
print(f"\n  AUC OOF:  {np.mean(fold_aucs):.4f}")
print(f"  KS:       {ks:.4f}")
print(f"  Gini:     {gini:.4f}")
print(f"  Umbral:   {best_thr:.3f}")
print(f"{'='*70}\n")
