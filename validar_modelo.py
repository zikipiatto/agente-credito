"""
Validación profunda del modelo LightGBM de decisión crediticia.
Métricas: PSI, sobreajuste, calibración, KS por decil, IV, estabilidad temporal,
          curvas de lift/ganancia, matriz de confusión por umbral.
"""

import warnings, json
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.metrics import (roc_auc_score, roc_curve, brier_score_loss,
                             confusion_matrix, precision_recall_curve)
from sklearn.calibration import calibration_curve
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

BASE_DIR = Path('/Users/fernandosierra/proyectos/agente-credito')
OUT_DIR  = BASE_DIR / 'modelo'
DATA_PATH = BASE_DIR / 'base_motor_decision_crediticia_2024_2025 1.csv'

print("=" * 70)
print("  VALIDACIÓN COMPLETA DEL MODELO")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════
# 1. CARGAR MODELO Y DATOS (misma limpieza que entrenar_modelo.py)
# ══════════════════════════════════════════════════════════════════
print("\n[1] Cargando modelo y datos...")

model       = joblib.load(OUT_DIR / 'lgbm_credito.pkl')
le_dict     = joblib.load(OUT_DIR / 'label_encoders.pkl')
feat_names  = joblib.load(OUT_DIR / 'feature_names.pkl')

with open(OUT_DIR / 'modelo_meta.json') as f:
    meta = json.load(f)

# Reproducir misma limpieza
df_raw = pd.read_csv(DATA_PATH, low_memory=False)
df = df_raw[df_raw['No_Credito'].notna() & df_raw['MaxAtr_ventana'].notna()].copy()
df['malo'] = (df['MaxAtr_ventana'] >= 30).astype(int)

# Guardar fecha para análisis temporal
df['fecha_dt'] = pd.to_datetime(df['fecha_solicitud'], dayfirst=True, errors='coerce')
df['mes_año']  = df['fecha_dt'].dt.to_period('M')

DROP_COLS = [
    'numero_solicitud','No_Credito','nombre_prospecto','rfc','curp',
    'fecha_nacimiento','folio_buro','nombre_negocio','fecha_solicitud',
    'score_verificacion_id','score_reconocimiento_facial','score_deteccion_vida',
    'score_validacion_gobierno','score_video_selfie','ingreso_neto','fico_score',
    'MaxAtr_ventana','malo','probabilidad_api',
    'fecha_dt','mes_año',
]
df_feat = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors='ignore')

SENTINEL_COLS = ['creditos_activos','saldo_actual','saldo_vencido','valor_score',
                 'score_no_hit','vanohi','icc']
for col in SENTINEL_COLS:
    if col in df_feat.columns:
        df_feat[col] = pd.to_numeric(df_feat[col], errors='coerce').replace([-100,-10,-1], np.nan)

if 'peor_atraso_dias' in df_feat.columns:
    df_feat['peor_atraso_dias'] = pd.to_numeric(
        df_feat['peor_atraso_dias'].astype(str).str.strip().replace('-100', np.nan),
        errors='coerce').replace(-100, np.nan)

if 'cuota_ingreso_pct' in df_feat.columns:
    df_feat['cuota_ingreso_pct'] = df_feat['cuota_ingreso_pct'].replace([np.inf,-np.inf], np.nan)
    df_feat['cuota_ingreso_pct'] = df_feat['cuota_ingreso_pct'].clip(upper=df_feat['cuota_ingreso_pct'].quantile(0.99))

if 'tipo_score' in df_feat.columns:
    def limpiar_tipo_score(v):
        v = str(v).strip()
        if v in ('Sin Dato','0','nan',''): return 'SIN_DATO'
        try: float(v); return 'BC_SCORE_NUM'
        except: return v.upper()
    df_feat['tipo_score'] = df_feat['tipo_score'].apply(limpiar_tipo_score)

if 'decision_ml' in df_feat.columns:
    df_feat['decision_ml'] = df_feat['decision_ml'].astype(str).str.strip().replace({'Sin Dato':'SIN_DATO','nan':'SIN_DATO'})

if 'decil_ml' in df_feat.columns:
    df_feat['decil_ml'] = pd.to_numeric(df_feat['decil_ml'].astype(str).replace('Sin Dato', np.nan), errors='coerce')

cat_cols = df_feat.select_dtypes(include='object').columns.tolist()
for col in cat_cols:
    df_feat[col] = df_feat[col].astype(str).fillna('MISSING')
    if col in le_dict:
        le = le_dict[col]
        known = set(le.classes_)
        df_feat[col] = df_feat[col].apply(lambda x: x if x in known else 'MISSING')
        if 'MISSING' not in known:
            le.classes_ = np.append(le.classes_, 'MISSING')
        df_feat[col] = le.transform(df_feat[col])

null_pct = df_feat.isnull().mean()
df_feat = df_feat.drop(columns=null_pct[null_pct > 0.80].index.tolist(), errors='ignore')
nunique  = df_feat.nunique()
df_feat  = df_feat.drop(columns=nunique[nunique <= 1].index.tolist(), errors='ignore')

# Ingeniería de features
df_feat['tiene_hit_buro'] = df['creditos_activos'].apply(lambda x: 0 if pd.isna(x) or x == -100 else 1)
if 'saldo_actual' in df_feat.columns and 'creditos_activos' in df_feat.columns:
    df_feat['ratio_saldo_creditos'] = (df_feat['saldo_actual'] / (df_feat['creditos_activos'] + 1)).clip(0, 1e6)
if 'saldo_vencido' in df_feat.columns:
    df_feat['tiene_vencido'] = (df['saldo_vencido'] > 0).astype(int)
if 'total_ingresos' in df_feat.columns:
    df_feat['total_ingresos'] = pd.to_numeric(df_feat['total_ingresos'], errors='coerce')
    df_feat['ingreso_bajo'] = (df_feat['total_ingresos'] < 5000).astype(int)
if 'total_ingresos' in df_feat.columns and 'total_egresos' in df_feat.columns:
    df_feat['total_egresos'] = pd.to_numeric(df_feat['total_egresos'], errors='coerce')
    df_feat['capacidad_real'] = (df_feat['total_ingresos'] - df_feat['total_egresos']).clip(-1e6, 1e6)

# Alinear columnas con las del modelo
for col in feat_names:
    if col not in df_feat.columns:
        df_feat[col] = np.nan
df_feat = df_feat[feat_names]

X = df_feat.values
y = df['malo'].values
proba = model.predict_proba(X)[:, 1]

print(f"    Registros: {len(y):,} | Malos: {y.sum():,} ({y.mean():.1%})")

# ══════════════════════════════════════════════════════════════════
# 2. SOBREAJUSTE — Train vs OOF vs Test
# ══════════════════════════════════════════════════════════════════
print("\n[2] Análisis de sobreajuste...")

auc_train = roc_auc_score(y, model.predict_proba(X)[:, 1])
auc_oof   = meta['oof_auc']
auc_test  = meta['test_auc']

# 5-fold para obtener AUC train vs val por fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_train_aucs, fold_val_aucs = [], []
params = meta['params'].copy()
params['n_estimators'] = 800

for tr_idx, val_idx in skf.split(X, y):
    m = lgb.LGBMClassifier(**params)
    m.fit(X[tr_idx], y[tr_idx],
          eval_set=[(X[val_idx], y[val_idx])],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    fold_train_aucs.append(roc_auc_score(y[tr_idx], m.predict_proba(X[tr_idx])[:, 1]))
    fold_val_aucs.append(roc_auc_score(y[val_idx],  m.predict_proba(X[val_idx])[:, 1]))

gap = np.mean(fold_train_aucs) - np.mean(fold_val_aucs)
print(f"    AUC Train (promedio fold): {np.mean(fold_train_aucs):.4f} ± {np.std(fold_train_aucs):.4f}")
print(f"    AUC Val   (OOF):           {np.mean(fold_val_aucs):.4f} ± {np.std(fold_val_aucs):.4f}")
print(f"    AUC completo (train full):  {auc_train:.4f}")
print(f"    Gap train-val:              {gap:.4f}  ({'⚠ SOBREAJUSTE' if gap > 0.05 else '✓ OK'})")

# ══════════════════════════════════════════════════════════════════
# 3. PSI — Estabilidad poblacional (train vs test temporal)
# ══════════════════════════════════════════════════════════════════
print("\n[3] PSI — Population Stability Index...")

def calc_psi(expected, actual, bins=10):
    """PSI entre dos distribuciones de probabilidad."""
    # Definir bins sobre expected
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 3:
        return np.nan, pd.DataFrame()

    def bucket(arr):
        counts, _ = np.histogram(arr, bins=breakpoints)
        pct = counts / len(arr)
        pct = np.where(pct == 0, 0.0001, pct)
        return pct

    exp_pct = bucket(expected)
    act_pct = bucket(actual)
    psi_vals = (act_pct - exp_pct) * np.log(act_pct / exp_pct)
    psi_total = np.sum(psi_vals)

    df_psi = pd.DataFrame({
        'bin': range(1, len(exp_pct)+1),
        'exp_pct': exp_pct,
        'act_pct': act_pct,
        'psi_component': psi_vals,
    })
    return psi_total, df_psi

# Dividir por tiempo: primera mitad vs segunda mitad
df_temp = df[['fecha_dt','malo']].copy()
df_temp['proba'] = proba
df_temp = df_temp.dropna(subset=['fecha_dt'])
df_temp = df_temp.sort_values('fecha_dt')

mid = len(df_temp) // 2
early = df_temp.iloc[:mid]['proba'].values
late  = df_temp.iloc[mid:]['proba'].values

psi_temporal, df_psi_temp = calc_psi(early, late)

# PSI también entre tipo_solicitud si existe
psi_label = '< 0.10 estable · 0.10-0.25 monitorear · > 0.25 inestable'
print(f"    PSI temporal (1a mitad vs 2a mitad): {psi_temporal:.4f}")
if psi_temporal < 0.10:
    estado_psi = '✓ ESTABLE'
elif psi_temporal < 0.25:
    estado_psi = '⚠ MONITOREAR'
else:
    estado_psi = '✗ INESTABLE'
print(f"    Estado: {estado_psi}  ({psi_label})")

# PSI por mes (distribución del score mes a mes vs primer mes)
df_temp['mes'] = df_temp['fecha_dt'].dt.to_period('M')
meses = sorted(df_temp['mes'].dropna().unique())
ref_mes = meses[0]
ref_dist = df_temp[df_temp['mes'] == ref_mes]['proba'].values

psi_por_mes = {}
for mes in meses[1:]:
    arr = df_temp[df_temp['mes'] == mes]['proba'].values
    if len(arr) < 30:
        continue
    psi_val, _ = calc_psi(ref_dist, arr)
    psi_por_mes[str(mes)] = psi_val

print(f"\n    PSI mensual (vs primer mes {ref_mes}):")
for mes, psi_val in list(psi_por_mes.items())[:12]:
    barra = '█' * min(int(psi_val * 100), 40)
    alerta = '✗' if psi_val > 0.25 else ('⚠' if psi_val > 0.10 else '✓')
    print(f"    {mes}:  {psi_val:.4f}  {alerta}  {barra}")

# ══════════════════════════════════════════════════════════════════
# 4. CALIBRACIÓN
# ══════════════════════════════════════════════════════════════════
print("\n[4] Calibración del modelo...")

fraction_of_pos, mean_predicted = calibration_curve(y, proba, n_bins=10)
brier = brier_score_loss(y, proba)
# Brier skill score (vs modelo naive = tasa_base)
tasa_base = y.mean()
brier_naive = tasa_base * (1 - tasa_base)
bss = 1 - brier / brier_naive

print(f"    Brier Score:        {brier:.4f}  (0=perfecto, {brier_naive:.4f}=naive)")
print(f"    Brier Skill Score:  {bss:.4f}  (>0 mejor que naive, >0.3 bueno)")
print(f"    Calibración (obs vs pred):")
for obs, pred in zip(fraction_of_pos, mean_predicted):
    diff = obs - pred
    barra = '→' if abs(diff) < 0.03 else ('↑' if diff > 0 else '↓')
    print(f"    pred={pred:.2f}  obs={obs:.2f}  {barra}  diff={diff:+.3f}")

# ══════════════════════════════════════════════════════════════════
# 5. KS DETALLADO POR DECIL
# ══════════════════════════════════════════════════════════════════
print("\n[5] KS detallado por decil...")

df_ks = pd.DataFrame({'proba': proba, 'malo': y})
df_ks['decil'] = pd.qcut(df_ks['proba'], q=10, labels=range(1,11), duplicates='drop')

# KS = max(tpr - fpr)
fpr_c, tpr_c, _ = roc_curve(y, proba)
ks_global = float(np.max(tpr_c - fpr_c))

tabla_decil = df_ks.groupby('decil').agg(
    n=('malo','count'), malos=('malo','sum'), buenos=('malo', lambda x: (x==0).sum()),
    prob_media=('proba','mean')
).sort_index(ascending=False)
tabla_decil['tasa_mala'] = tabla_decil['malos'] / tabla_decil['n']
tabla_decil['pct_malos_cap'] = tabla_decil['malos'].cumsum() / tabla_decil['malos'].sum()
tabla_decil['pct_buenos_cap'] = tabla_decil['buenos'].cumsum() / tabla_decil['buenos'].sum()
tabla_decil['lift'] = tabla_decil['tasa_mala'] / y.mean()

print(f"\n    {'Decil':>5} {'N':>6} {'Malos':>6} {'Tasa%':>7} {'Lift':>6} {'%Malos cap':>11} {'%Buenos cap':>12}")
print(f"    {'─'*60}")
for dec, row in tabla_decil.iterrows():
    print(f"    {dec:>5} {row['n']:>6,} {row['malos']:>6,} {row['tasa_mala']*100:>6.1f}% "
          f"{row['lift']:>6.2f}x {row['pct_malos_cap']*100:>10.1f}% {row['pct_buenos_cap']*100:>11.1f}%")

print(f"\n    KS global: {ks_global:.4f}")

# ══════════════════════════════════════════════════════════════════
# 6. INFORMATION VALUE (IV) POR FEATURE
# ══════════════════════════════════════════════════════════════════
print("\n[6] Information Value (IV) por feature...")

def calc_iv(feature_vals, target, bins=10):
    """IV de una variable numérica vs target binario."""
    df_iv = pd.DataFrame({'x': feature_vals, 'y': target}).dropna()
    if df_iv['x'].nunique() < 2:
        return 0.0
    try:
        df_iv['bin'] = pd.qcut(df_iv['x'], q=bins, duplicates='drop')
    except:
        return 0.0

    total_bad  = (df_iv['y'] == 1).sum()
    total_good = (df_iv['y'] == 0).sum()
    if total_bad == 0 or total_good == 0:
        return 0.0

    grp = df_iv.groupby('bin')['y'].agg(['sum','count'])
    grp.columns = ['bad','total']
    grp['good'] = grp['total'] - grp['bad']
    grp['pct_bad']  = grp['bad']  / total_bad
    grp['pct_good'] = grp['good'] / total_good
    grp = grp[(grp['pct_bad'] > 0) & (grp['pct_good'] > 0)]
    grp['woe'] = np.log(grp['pct_bad'] / grp['pct_good'])
    grp['iv']  = (grp['pct_bad'] - grp['pct_good']) * grp['woe']
    return grp['iv'].sum()

iv_results = []
for i, col in enumerate(feat_names):
    iv = calc_iv(X[:, i], y)
    iv_results.append({'feature': col, 'iv': iv})

df_iv = pd.DataFrame(iv_results).sort_values('iv', ascending=False)
df_iv['poder'] = df_iv['iv'].apply(lambda v:
    'Muy fuerte' if v > 0.5 else ('Fuerte' if v > 0.3 else ('Medio' if v > 0.1 else
    ('Débil' if v > 0.02 else 'Sin poder'))))

print(f"\n    {'Feature':<30} {'IV':>8}  Poder")
print(f"    {'─'*55}")
for _, row in df_iv.iterrows():
    print(f"    {row['feature']:<30} {row['iv']:>8.4f}  {row['poder']}")

weak_feats = df_iv[df_iv['iv'] < 0.02]['feature'].tolist()
print(f"\n    Features con IV < 0.02 (considerar eliminar): {weak_feats}")

# ══════════════════════════════════════════════════════════════════
# 7. ANÁLISIS DE UMBRALES — precisión, recall, F1
# ══════════════════════════════════════════════════════════════════
print("\n[7] Análisis de umbrales de decisión...")

prec_arr, rec_arr, thr_arr = precision_recall_curve(y, proba)
f1_arr = 2 * prec_arr[:-1] * rec_arr[:-1] / (prec_arr[:-1] + rec_arr[:-1] + 1e-9)

print(f"\n    {'Umbral':>8} {'Prec%':>7} {'Recall%':>8} {'F1':>7} {'FP':>8} {'FN':>8} {'Correctos%':>11}")
print(f"    {'─'*68}")
for thr in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]:
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    prec  = tp / (tp + fp) if (tp+fp) > 0 else 0
    rec   = tp / (tp + fn) if (tp+fn) > 0 else 0
    f1    = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0
    acc   = (tp+tn) / len(y)
    print(f"    {thr:>8.2f} {prec*100:>6.1f}% {rec*100:>7.1f}% {f1:>7.3f} {fp:>8,} {fn:>8,} {acc*100:>10.1f}%")

# ══════════════════════════════════════════════════════════════════
# 8. ESTABILIDAD TEMPORAL DEL AUC
# ══════════════════════════════════════════════════════════════════
print("\n[8] Estabilidad temporal del AUC por mes...")

df_temp2 = df[['fecha_dt','malo']].copy()
df_temp2['proba'] = proba
df_temp2['mes'] = df_temp2['fecha_dt'].dt.to_period('M')

auc_por_mes = {}
for mes in sorted(df_temp2['mes'].dropna().unique()):
    sub = df_temp2[df_temp2['mes'] == mes]
    if sub['malo'].sum() < 10 or (sub['malo'] == 0).sum() < 10:
        continue
    try:
        auc_m = roc_auc_score(sub['malo'], sub['proba'])
        auc_por_mes[str(mes)] = auc_m
    except:
        pass

print(f"\n    {'Mes':<10} {'AUC':>7}  Estabilidad")
print(f"    {'─'*40}")
auc_vals = list(auc_por_mes.values())
auc_media = np.mean(auc_vals) if auc_vals else 0
for mes, auc_m in auc_por_mes.items():
    diff = auc_m - auc_media
    estado = '✓' if abs(diff) < 0.03 else ('⚠' if abs(diff) < 0.06 else '✗')
    print(f"    {mes:<10} {auc_m:.4f}  {estado}  {diff:+.4f} vs media")

if auc_vals:
    print(f"\n    AUC media mensual: {auc_media:.4f}")
    print(f"    Desv. estándar:    {np.std(auc_vals):.4f}  ({'✓ estable' if np.std(auc_vals) < 0.03 else '⚠ variabilidad'})")

# ══════════════════════════════════════════════════════════════════
# 9. RESUMEN EJECUTIVO DE VALIDACIÓN
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  RESUMEN EJECUTIVO DE VALIDACIÓN")
print("=" * 70)

checks = [
    ('AUC OOF',           f"{meta['oof_auc']:.4f}",          meta['oof_auc'] >= 0.75,      '≥ 0.75'),
    ('KS',                f"{meta['ks']:.4f}",                meta['ks'] >= 0.40,           '≥ 0.40'),
    ('Gini',              f"{meta['gini']:.4f}",              meta['gini'] >= 0.60,         '≥ 0.60'),
    ('Gap sobreajuste',   f"{gap:.4f}",                       gap < 0.05,                   '< 0.05'),
    ('PSI temporal',      f"{psi_temporal:.4f}",              psi_temporal < 0.10,          '< 0.10'),
    ('Brier Skill Score', f"{bss:.4f}",                       bss >= 0.10,                  '≥ 0.10'),
    ('AUC estd mensual',  f"{np.std(auc_vals):.4f}" if auc_vals else 'N/A',
                          np.std(auc_vals) < 0.03 if auc_vals else False,                   '< 0.03'),
]

for nombre, valor, ok, umbral in checks:
    estado = '✅ OK' if ok else '⚠ REVISAR'
    print(f"  {nombre:<22} {valor:>8}   {estado:12}  (referencia {umbral})")

print("=" * 70)

# ══════════════════════════════════════════════════════════════════
# 10. GRÁFICAS
# ══════════════════════════════════════════════════════════════════
print("\n[10] Generando gráficas...")

fig = plt.figure(figsize=(18, 22))
gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)

AZUL  = '#1a73e8'
VERDE = '#34a853'
ROJO  = '#ea4335'
NARJ  = '#fbbc04'

# ── 1. ROC curve
ax = fig.add_subplot(gs[0, 0])
fpr_c, tpr_c, _ = roc_curve(y, proba)
ax.plot(fpr_c, tpr_c, color=AZUL, lw=2, label=f'AUC={meta["oof_auc"]:.3f}')
ax.fill_between(fpr_c, tpr_c, alpha=0.08, color=AZUL)
ax.plot([0,1],[0,1],'k--',lw=1)
ax.set_title('Curva ROC', fontweight='bold')
ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
ax.legend(); ax.grid(alpha=0.3)

# ── 2. Precision-Recall
ax = fig.add_subplot(gs[0, 1])
ax.plot(rec_arr, prec_arr, color=VERDE, lw=2)
ax.axhline(y.mean(), color='k', ls='--', lw=1, label=f'Baseline={y.mean():.2f}')
ax.set_title('Curva Precision-Recall', fontweight='bold')
ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.legend(); ax.grid(alpha=0.3)

# ── 3. Calibración
ax = fig.add_subplot(gs[0, 2])
ax.plot(mean_predicted, fraction_of_pos, 'o-', color=AZUL, lw=2, label='Modelo')
ax.plot([0,1],[0,1],'k--',lw=1, label='Perfecta')
ax.set_title(f'Calibración (Brier={brier:.3f})', fontweight='bold')
ax.set_xlabel('Probabilidad predicha'); ax.set_ylabel('Fracción real de malos')
ax.legend(); ax.grid(alpha=0.3)

# ── 4. Distribución score por bueno/malo
ax = fig.add_subplot(gs[1, 0])
ax.hist(proba[y==0], bins=50, alpha=0.6, color=VERDE, label='Buenos', density=True)
ax.hist(proba[y==1], bins=50, alpha=0.6, color=ROJO,  label='Malos',  density=True)
ax.axvline(meta['umbral_decision'], color='k', ls='--', lw=1.5, label=f'Umbral={meta["umbral_decision"]:.2f}')
ax.set_title('Distribución de scores', fontweight='bold')
ax.set_xlabel('Probabilidad predicha'); ax.set_ylabel('Densidad')
ax.legend(); ax.grid(alpha=0.3)

# ── 5. Lift chart
ax = fig.add_subplot(gs[1, 1])
lift_vals = tabla_decil['lift'].sort_index().values
deciles   = list(range(1, len(lift_vals)+1))
bars = ax.bar(deciles, lift_vals, color=[ROJO if l > 2 else (NARJ if l > 1 else VERDE) for l in lift_vals])
ax.axhline(1, color='k', ls='--', lw=1)
ax.set_title('Lift por decil', fontweight='bold')
ax.set_xlabel('Decil (1=mejor cliente)'); ax.set_ylabel('Lift')
ax.set_xticks(deciles); ax.grid(alpha=0.3, axis='y')

# ── 6. Gain chart
ax = fig.add_subplot(gs[1, 2])
pct_pob  = np.cumsum([1/10]*10)
pct_malos_cap = tabla_decil['pct_malos_cap'].sort_index().values
ax.plot(pct_pob, pct_malos_cap, 'o-', color=AZUL, lw=2, label='Modelo')
ax.plot([0,1],[0,1],'k--',lw=1, label='Aleatorio')
ax.plot([y.mean(),1],[1,1],'g--',lw=1, alpha=0.5, label='Perfecto')
ax.set_title('Curva de Ganancia', fontweight='bold')
ax.set_xlabel('% Población'); ax.set_ylabel('% Malos capturados')
ax.legend(); ax.grid(alpha=0.3)

# ── 7. PSI por mes
ax = fig.add_subplot(gs[2, 0])
meses_plot = list(psi_por_mes.keys())[:18]
psi_vals_plot = [psi_por_mes[m] for m in meses_plot]
colors_psi = [ROJO if v > 0.25 else (NARJ if v > 0.10 else VERDE) for v in psi_vals_plot]
ax.bar(range(len(meses_plot)), psi_vals_plot, color=colors_psi)
ax.axhline(0.10, color=NARJ, ls='--', lw=1, label='Límite monitoreo')
ax.axhline(0.25, color=ROJO, ls='--', lw=1, label='Límite inestabilidad')
ax.set_xticks(range(len(meses_plot)))
ax.set_xticklabels(meses_plot, rotation=45, ha='right', fontsize=7)
ax.set_title('PSI mensual vs mes base', fontweight='bold')
ax.set_ylabel('PSI'); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

# ── 8. AUC mensual
ax = fig.add_subplot(gs[2, 1])
meses_auc = list(auc_por_mes.keys())
auc_auc   = list(auc_por_mes.values())
ax.plot(range(len(meses_auc)), auc_auc, 'o-', color=AZUL, lw=2)
ax.axhline(auc_media, color='k', ls='--', lw=1, label=f'Media={auc_media:.3f}')
ax.axhspan(auc_media-0.03, auc_media+0.03, alpha=0.1, color=VERDE, label='±0.03')
ax.set_xticks(range(len(meses_auc)))
ax.set_xticklabels(meses_auc, rotation=45, ha='right', fontsize=7)
ax.set_title('AUC mensual (estabilidad temporal)', fontweight='bold')
ax.set_ylabel('AUC'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

# ── 9. Train vs Val AUC por fold
ax = fig.add_subplot(gs[2, 2])
folds = list(range(1, 6))
ax.plot(folds, fold_train_aucs, 'o-', color=VERDE, lw=2, label='Train')
ax.plot(folds, fold_val_aucs,   'o-', color=AZUL,  lw=2, label='Validación (OOF)')
ax.fill_between(folds, fold_train_aucs, fold_val_aucs, alpha=0.15, color=NARJ, label=f'Gap={gap:.3f}')
ax.set_title('Sobreajuste: Train vs Val por fold', fontweight='bold')
ax.set_xlabel('Fold'); ax.set_ylabel('AUC')
ax.legend(); ax.grid(alpha=0.3)
ax.set_xticks(folds)

# ── 10. IV por feature
ax = fig.add_subplot(gs[3, :2])
df_iv_plot = df_iv.head(20).sort_values('iv')
colors_iv = [ROJO if v > 0.5 else (NARJ if v > 0.3 else (AZUL if v > 0.1 else '#ccc')) for v in df_iv_plot['iv']]
ax.barh(df_iv_plot['feature'], df_iv_plot['iv'], color=colors_iv)
ax.axvline(0.02, color='k', ls='--', lw=1, label='Min recomendado (0.02)')
ax.axvline(0.10, color=NARJ, ls='--', lw=1, label='Poder medio (0.10)')
ax.axvline(0.30, color=VERDE, ls='--', lw=1, label='Poder fuerte (0.30)')
ax.set_title('Information Value (IV) por feature', fontweight='bold')
ax.set_xlabel('IV'); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='x')

# ── 11. Resumen visual
ax = fig.add_subplot(gs[3, 2])
ax.axis('off')
resumen_txt = [
    ('RESUMEN DE VALIDACIÓN', 14, 'bold', '#1a1a2e'),
    ('', 10, 'normal', 'white'),
]
for nombre, valor, ok, _ in checks:
    icono = '✅' if ok else '⚠️'
    resumen_txt.append((f'{icono} {nombre}: {valor}', 9.5, 'normal', '#333'))

y_pos = 0.97
for txt, fsize, fweight, fcolor in resumen_txt:
    ax.text(0.05, y_pos, txt, transform=ax.transAxes,
            fontsize=fsize, fontweight=fweight, color=fcolor,
            verticalalignment='top')
    y_pos -= 0.085

plt.suptitle('Validación Completa — Modelo LightGBM Crediticio',
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig(OUT_DIR / 'validacion_completa.png', dpi=150, bbox_inches='tight')
plt.close()

# Guardar reporte de validación
reporte = {
    'auc_oof':          meta['oof_auc'],
    'auc_test':         meta['test_auc'],
    'ks':               meta['ks'],
    'gini':             meta['gini'],
    'gap_sobreajuste':  float(gap),
    'psi_temporal':     float(psi_temporal),
    'psi_estado':       estado_psi,
    'brier_score':      float(brier),
    'brier_skill':      float(bss),
    'auc_mensual_media': float(auc_media) if auc_vals else None,
    'auc_mensual_std':  float(np.std(auc_vals)) if auc_vals else None,
    'auc_por_mes':      auc_por_mes,
    'psi_por_mes':      psi_por_mes,
    'iv_features':      df_iv[['feature','iv','poder']].to_dict('records'),
    'features_debiles': weak_feats,
}
with open(OUT_DIR / 'reporte_validacion.json', 'w') as f:
    json.dump(reporte, f, indent=2, ensure_ascii=False, default=str)

print(f"\n  Gráficas: {OUT_DIR}/validacion_completa.png")
print(f"  Reporte:  {OUT_DIR}/reporte_validacion.json")
print("\n  ✅ Validación completa.\n")
