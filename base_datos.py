"""
Módulo de persistencia — SQLite local
Guarda casos resueltos y feedback del analista para retroalimentar patrones.
"""
import sqlite3, os, json
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "casos_credito.db")

def init_db():
    """Crea las tablas si no existen."""
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS casos (
                folio TEXT PRIMARY KEY,
                fecha TEXT,
                decision_sistema TEXT,
                score_sistema INTEGER,
                bc_score REAL,
                icc INTEGER,
                decil_ml INTEGER,
                nivel_endeudamiento TEXT,
                ratio_cuota_ingreso REAL,
                total_cuentas INTEGER,
                peor_mop INTEGER,
                tiene_juicios INTEGER,
                creditos_vencidos INTEGER,
                monto REAL,
                producto TEXT,
                resumen_caso TEXT,
                deliberacion_ia TEXT,
                decision_analista TEXT DEFAULT NULL,
                comentario_analista TEXT DEFAULT NULL,
                fecha_feedback TEXT DEFAULT NULL
            )
        """)
        con.commit()

def guardar_caso(folio: str, decision_sistema: str, score_sistema: int,
                 datos: dict, resumen_caso: str, deliberacion_ia: str):
    """Guarda un caso analizado en la BD."""
    init_db()
    buro = datos.get("buro", {})
    ab   = datos.get("analisis_buro", {})
    fin  = datos.get("resultado_financiero", {})
    cond = datos.get("condiciones", {})

    bc_score_raw = buro.get("score")
    try:
        bc_score = float(bc_score_raw) if bc_score_raw else None
    except:
        bc_score = None

    icc_raw = buro.get("icc")
    try:
        icc = int(str(icc_raw).strip()) if icc_raw else None
    except:
        icc = None

    ml = datos.get("modelo_ml") or {}
    try:
        decil_ml = int(ml.get("valor_decil")) if ml.get("valor_decil") else None
    except:
        decil_ml = None

    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
            INSERT OR REPLACE INTO casos
            (folio, fecha, decision_sistema, score_sistema, bc_score, icc, decil_ml,
             nivel_endeudamiento, ratio_cuota_ingreso, total_cuentas, peor_mop,
             tiene_juicios, creditos_vencidos, monto, producto, resumen_caso, deliberacion_ia)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            folio,
            datetime.now().isoformat(),
            decision_sistema,
            score_sistema,
            bc_score,
            icc,
            decil_ml,
            ab.get("nivel_endeudamiento"),
            fin.get("ratio_cuota_ingreso"),
            ab.get("total_cuentas", 0),
            ab.get("peor_mop_historico", 0),
            1 if buro.get("tiene_juicios") else 0,
            buro.get("creditos_vencidos", 0),
            cond.get("monto"),
            cond.get("producto"),
            resumen_caso,
            deliberacion_ia,
        ))
        con.commit()

def registrar_feedback(folio: str, decision_analista: str, comentario: str):
    """Registra la decisión y comentario del analista de mesa de crédito."""
    init_db()
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
            UPDATE casos SET decision_analista=?, comentario_analista=?, fecha_feedback=?
            WHERE folio=?
        """, (decision_analista, comentario, datetime.now().isoformat(), folio))
        con.commit()
    return con.total_changes

def buscar_casos_similares(bc_score: float | None, icc: int | None,
                           score_sistema: int, decision_sistema: str, n: int = 3) -> list[dict]:
    """
    Busca casos similares con feedback del analista.
    Filtros: score±2, bc_score±100 (si disponible), con feedback registrado.
    """
    init_db()
    params = [score_sistema - 2, score_sistema + 2]
    query = """
        SELECT folio, fecha, decision_sistema, score_sistema, bc_score, icc,
               nivel_endeudamiento, ratio_cuota_ingreso, total_cuentas, peor_mop,
               decision_analista, comentario_analista, resumen_caso
        FROM casos
        WHERE decision_analista IS NOT NULL
          AND score_sistema BETWEEN ? AND ?
    """
    if bc_score is not None:
        query += " AND (bc_score IS NULL OR ABS(bc_score - ?) <= 100)"
        params.append(bc_score)
    query += " ORDER BY fecha DESC LIMIT ?"
    params.append(n)

    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(query, params).fetchall()
    return [dict(r) for r in rows]

def listar_casos(limit: int = 50) -> list[dict]:
    """Lista los últimos N casos para el dashboard."""
    init_db()
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute("""
            SELECT folio, fecha, decision_sistema, score_sistema, bc_score, icc,
                   decision_analista, comentario_analista, fecha_feedback, monto, producto
            FROM casos ORDER BY fecha DESC LIMIT ?
        """, (limit,)).fetchall()
    return [dict(r) for r in rows]
