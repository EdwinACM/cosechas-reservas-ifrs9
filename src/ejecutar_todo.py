"""Corre todos los pasos en orden. Con la misma semilla da los mismos resultados."""
import subprocess
import sys
from pathlib import Path

PASOS = [
    "paso0_generar_datos.py",
    "paso1_cosechas.py",
    "paso2_transiciones.py",
    "paso3_pd.py",
    "paso4_lgd_ead.py",
    "paso5_reserva.py",
    "paso6_sensibilidad.py",
]

if __name__ == "__main__":
    carpeta = Path(__file__).resolve().parent
    for paso in PASOS:
        print(f"\n=== {paso} ===", flush=True)
        subprocess.run([sys.executable, str(carpeta / paso)], check=True, cwd=carpeta)
