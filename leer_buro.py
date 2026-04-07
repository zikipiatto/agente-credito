"""
Lectura de reportes de Buró de Crédito PF — versión individual y por lotes.

Uso rápido:
    from leer_buro import leer_buro, leer_buro_lote

    # Un solo PDF
    datos = leer_buro("reporte.pdf")

    # Directorio completo → DataFrame
    df = leer_buro_lote("Ejemplos Solicitud y Buro/Reporte de Buro/")

    # Lista explícita de rutas → DataFrame
    df = leer_buro_lote(["a.pdf", "b.pdf", "c.pdf"])

Desde terminal:
    python leer_buro.py ruta/al/directorio/      # lote → CSV
    python leer_buro.py reporte.pdf              # uno → imprime dict
"""
from __future__ import annotations

import sys
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Union

import pandas as pd

from parsear_buro import parsear_pdf_buro


# ─────────────────────────────────────────────────────────────────────────────
# Versión individual
# ─────────────────────────────────────────────────────────────────────────────

def leer_buro(ruta: Union[str, Path]) -> dict:
    """
    Lee un solo PDF de Buró de Crédito y devuelve un diccionario con todos
    los campos parseados, incluyendo el texto raw para debugging.

    Args:
        ruta: ruta al archivo PDF (str o Path)

    Returns:
        dict con: score, icc, tipo_score, causas_score, alertas_hawk,
                  tiene_juicios, cuentas_abiertas, cuentas_cerradas,
                  creditos_vencidos, saldo_actual, saldo_vencido,
                  pago_a_realizar, peor_atraso_dias, cuentas (list),
                  rfc, nombre, archivo, _texto_raw
    """
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(f"No existe el archivo: {ruta}")
    if ruta.suffix.lower() != ".pdf":
        raise ValueError(f"El archivo no es un PDF: {ruta.name}")

    datos = parsear_pdf_buro(str(ruta))
    datos["archivo"] = ruta.name
    datos["ruta"] = str(ruta)
    return datos


# ─────────────────────────────────────────────────────────────────────────────
# Versión por lotes
# ─────────────────────────────────────────────────────────────────────────────

def leer_buro_lote(
    fuente: Union[str, Path, list],
    *,
    recursivo: bool = True,
    workers: int = 4,
    incluir_cuentas: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Lee múltiples PDFs de Buró de Crédito y devuelve un DataFrame.

    Cada fila = un PDF. Las columnas son los campos planos del buró
    (sin _texto_raw ni la lista de cuentas, para mantener el DataFrame manejable).

    Args:
        fuente:          directorio (str/Path), o lista de rutas (str/Path)
        recursivo:       si fuente es directorio, busca PDFs en subdirectorios
        workers:         hilos en paralelo (I/O bound; 4 es suficiente)
        incluir_cuentas: si True, agrega columna 'cuentas' con la lista de dicts
        verbose:         imprime progreso y errores en pantalla

    Returns:
        pd.DataFrame con columnas:
            archivo, ruta, nombre, rfc,
            score, icc, tipo_score,
            cuentas_abiertas, cuentas_cerradas, creditos_vencidos,
            saldo_actual, saldo_vencido, pago_a_realizar, peor_atraso_dias,
            tiene_juicios, n_alertas_hawk, n_causas_score,
            n_cuentas_detalle, [cuentas si incluir_cuentas=True],
            error
    """
    # ── Resolver lista de archivos ──
    rutas = _resolver_rutas(fuente, recursivo)
    if not rutas:
        raise FileNotFoundError(f"No se encontraron PDFs en: {fuente}")

    if verbose:
        print(f"Procesando {len(rutas)} PDFs con {workers} workers...")

    # ── Procesamiento paralelo ──
    resultados = []
    errores = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futuros = {executor.submit(_parsear_seguro, r): r for r in rutas}

        for i, futuro in enumerate(as_completed(futuros), 1):
            ruta = futuros[futuro]
            fila = futuro.result()
            resultados.append(fila)

            if verbose:
                estado = "OK" if fila["error"] is None else f"ERROR: {fila['error']}"
                print(f"  [{i:>3}/{len(rutas)}] {Path(ruta).name[:55]:<55} {estado}")
            if fila["error"]:
                errores += 1

    if verbose:
        ok = len(resultados) - errores
        print(f"\nListo: {ok} exitosos, {errores} errores de {len(rutas)} PDFs.")

    # ── Construir DataFrame ──
    df = pd.DataFrame(_aplanar(r, incluir_cuentas) for r in resultados)

    # Ordenar por nombre de archivo
    df = df.sort_values("archivo").reset_index(drop=True)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Helpers internos
# ─────────────────────────────────────────────────────────────────────────────

def _resolver_rutas(fuente, recursivo: bool) -> list[Path]:
    """Convierte fuente (dir / lista) en lista de Path a PDFs."""
    if isinstance(fuente, (list, tuple)):
        return [Path(r) for r in fuente]

    fuente = Path(fuente)
    if fuente.is_file():
        return [fuente]
    if fuente.is_dir():
        patron = "**/*.pdf" if recursivo else "*.pdf"
        return sorted(fuente.glob(patron))

    raise ValueError(f"'fuente' no es un archivo, directorio ni lista: {fuente}")


def _parsear_seguro(ruta: Path) -> dict:
    """Llama al parser capturando excepciones para no romper el lote."""
    try:
        datos = parsear_pdf_buro(str(ruta))
        datos["archivo"] = ruta.name
        datos["ruta"] = str(ruta)
        datos["error"] = None
        return datos
    except Exception as e:
        return {
            "archivo": ruta.name,
            "ruta": str(ruta),
            "error": str(e),
        }


def _aplanar(datos: dict, incluir_cuentas: bool) -> dict:
    """Convierte el dict del parser en una fila plana para el DataFrame."""
    fila = {
        "archivo":           datos.get("archivo"),
        "ruta":              datos.get("ruta"),
        "nombre":            datos.get("nombre"),
        "rfc":               datos.get("rfc"),
        "folio":             datos.get("folio"),
        "fecha_consulta":    datos.get("fecha_consulta"),
        "score":             datos.get("score"),
        "icc":               datos.get("icc"),
        "tipo_score":        datos.get("tipo_score"),
        "cuentas_abiertas":  datos.get("cuentas_abiertas"),
        "cuentas_cerradas":  datos.get("cuentas_cerradas"),
        "creditos_vencidos": datos.get("creditos_vencidos"),
        "saldo_actual":      datos.get("saldo_actual"),
        "saldo_vencido":     datos.get("saldo_vencido"),
        "pago_a_realizar":   datos.get("pago_a_realizar"),
        "peor_atraso_dias":  datos.get("peor_atraso_dias"),
        "tiene_juicios":     datos.get("tiene_juicios"),
        "n_alertas_hawk":    len(datos.get("alertas_hawk") or []),
        "n_causas_score":    len(datos.get("causas_score") or []),
        "n_cuentas_detalle": len(datos.get("cuentas") or []),
        "error":             datos.get("error"),
    }
    if incluir_cuentas:
        fila["cuentas"] = datos.get("cuentas")
    return fila


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    objetivo = Path(sys.argv[1])

    if objetivo.is_file() and objetivo.suffix.lower() == ".pdf":
        # Modo individual — imprime el dict (sin texto raw)
        datos = leer_buro(objetivo)
        datos.pop("_texto_raw", None)
        datos.pop("cuentas", None)
        print(json.dumps(datos, ensure_ascii=False, indent=2, default=str))

    elif objetivo.is_dir():
        # Modo lote — guarda CSV
        df = leer_buro_lote(objetivo)
        salida = objetivo / "buro_parseado.csv"
        df.to_csv(salida, index=False)
        print(f"\nCSV guardado en: {salida}")
        print(df[["archivo", "nombre", "score", "icc", "cuentas_abiertas",
                   "creditos_vencidos", "saldo_vencido", "error"]].to_string(index=False))

    else:
        print(f"ERROR: '{objetivo}' no es un PDF ni un directorio.")
        sys.exit(1)
