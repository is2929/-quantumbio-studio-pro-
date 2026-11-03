"""
Punto de entrada del servidor Flask de "QuantumBio Studio Pro".

Este módulo se encarga de:
    - Crear y configurar la aplicación Flask.
    - Registrar los "blueprints" (módulos) científicos disponibles
      (Química Computacional y Bioinformática).
    - Exponer una serie de endpoints REST (/api/...) usados por la
      interfaz web para:
        * Consultar información del sistema (CPU/RAM).
        * Recomendar parámetros de cálculo según la tarea y los
          recursos disponibles (asistente "AI").
        * Cancelar trabajos en ejecución.
        * Consultar el estado de un trabajo.
        * Generar reportes en texto plano de resultados.
        * Servir ficheros generados dentro de la carpeta de trabajos.
    - Servir la página principal (index.html).

Arquitectura general
--------------------
    - ``core`` expone utilidades compartidas (rutas, locks, procesos
      activos, diccionario de trabajos, flags de disponibilidad, logger,
      etc.). Se importa tanto el módulo como sus nombres (``from core import *``)
      para que el resto del código pueda usarlos directamente.
    - Cada módulo científico registra su propio Blueprint, lo cual
      permite añadir/quitar funcionalidades sin tocar este archivo.
"""

import os
from datetime import datetime

from flask import Flask, request, jsonify, render_template, send_from_directory
import psutil

# ``core`` contiene configuración global compartida entre todos los módulos:
#   - JOBS: diccionario con los trabajos en curso/finalizados.
#   - JOBS_DIR: carpeta raíz donde se guardan los ficheros por trabajo.
#   - JOBS_LOCK: lock de sincronización para acceso concurrente a JOBS.
#   - ACTIVE_PROCESSES: procesos externos lanzados por trabajo (para cancelar).
#   - BSE_AVAILABLE, OPENMMFF_AVAILABLE: flags de dependencias opcionales.
#   - logger: logger central de la aplicación.
import core
from core import *


# --------------------------------------------------------------------------- #
# Configuración de la aplicación Flask
# --------------------------------------------------------------------------- #
app = Flask(__name__, template_folder="templates", static_folder="static")

# Tamaño máximo permitido para subidas (64 MB). Evita abusos de memoria
# al recibir estructuras moleculares o trayectorias.
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024


# --------------------------------------------------------------------------- #
# Referencias bibliográficas de los paquetes científicos de terceros
# --------------------------------------------------------------------------- #
# Este diccionario centraliza las citas de todos los paquetes que utiliza
# la plataforma. Se usan al generar los reportes para dar crédito a las
# herramientas de las que dependemos (PySCF, ORCA, OpenMM, RDKit, etc.).
#
# Cada entrada es un string ya formateado (autores, título, revista, año,
# DOI o URL) que se imprime como una línea del bloque REFERENCES.
#
# Para añadir una nueva cita:
#   1. Agrega una clave nueva en este diccionario.
#   2. Inclúyela en el conjunto ``refs`` correspondiente dentro de
#      ``generate_report`` (según el tipo de cálculo y el motor usado).
CITATIONS = {
    "pyscf": (
        "Sun, Q.; Zhang, X.; Banerjee, S.; Bao, P.; Barbry, M.; Blunt, N. S.; "
        "Bogdanov, N. A.; Booth, G. H.; Chen, J.; Cui, Y.; et al. Recent developments "
        "in the PySCF program package. J. Chem. Phys. 2020, 153 (2), 024109. "
        "DOI: 10.1063/5.0006074"
    ),
    "orca": (
        "Neese, F. Software update: The ORCA program system—Version 5.0. "
        "WIREs Comput. Mol. Sci. 2022, 12 (5), e1606. DOI: 10.1002/wcms.1606"
    ),
    "openmm": (
        "Eastman, P.; Swails, J.; Chodera, J. D.; McGibbon, R. T.; Zhao, Y.; "
        "Beauchamp, K. A.; Wang, L.-P.; Simmonett, A. C.; Harrigan, M. P.; "
        "Stern, C. D.; et al. OpenMM 7: Rapid development of high performance "
        "algorithms for molecular dynamics. PLOS Comput. Biol. 2017, 13 (7), "
        "e1005659. DOI: 10.1371/journal.pcbi.1005659"
    ),
    "openff": (
        "Qiu, Y.; Smith, D. G. A.; Boothroyd, S.; Jang, H.; Hahn, D. F.; Wagner, J.; "
        "Bannan, C. C.; Gokey, T.; Lim, V. T.; Stern, C. D.; et al. Development and "
        "Benchmarking of Open Force Field v1.0.0—the Parsley Small-Molecule Force "
        "Field. J. Chem. Theory Comput. 2021, 17 (10), 6262–6280. "
        "DOI: 10.1021/acs.jctc.1c00571"
    ),
    "vina": (
        "Trott, O.; Olson, A. J. AutoDock Vina: Improving the speed and accuracy of "
        "docking with a new scoring function, efficient optimization, and "
        "multithreading. J. Comput. Chem. 2010, 31 (2), 455–461. "
        "DOI: 10.1002/jcc.21334"
    ),
    "alphafold": (
        "Jumper, J.; Evans, R.; Pritzel, A.; Green, T.; Figurnov, M.; Ronneberger, O.; "
        "Tunyasuvunakool, K.; Bates, R.; Žídek, A.; Potapenko, A.; et al. Highly "
        "accurate protein structure prediction with AlphaFold. Nature 2021, 596 "
        "(7873), 583–589. DOI: 10.1038/s41586-021-03819-2"
    ),
    "esmfold": (
        "Lin, Z.; Akin, H.; Rao, R.; Hie, B.; Zhu, Z.; Lu, W.; Smetanin, N.; "
        "Verkuil, R.; Kabeli, O.; Shmueli, Y.; et al. Evolutionary-scale prediction "
        "of atomic-level protein structure with a language model. Science 2023, "
        "379 (6637), 1123–1130. DOI: 10.1126/science.ade2574"
    ),
    "pdbfixer": (
        "Eastman, P. PDBFixer; https://github.com/openmm/pdbfixer (accessed 2024). "
        "PDBFixer fixes problems in PDB files."
    ),
    "rdkit": (
        "RDKit: Open-source cheminformatics; https://www.rdkit.org (no official "
        "publication)."
    ),
    "bse": (
        "Pritchard, B. P.; Altarawy, D.; Didier, B.; Gibson, T. D.; Windus, T. L. "
        "New Basis Set Exchange: An Open, Up-to-Date Resource for the Molecular "
        "Sciences Community. J. Chem. Inf. Model. 2019, 59 (11), 4814–4820. "
        "DOI: 10.1021/acs.jcim.9b00725"
    ),
    "pymol": (
        "Schrödinger, LLC. The PyMOL Molecular Graphics System, Version 2.0; "
        "Schrödinger, LLC: New York, 2015."
    ),
    "openbabel": (
        "O'Boyle, N. M.; Banck, M.; James, C. A.; Morley, C.; Vandermeersch, T.; "
        "Hutchison, G. R. Open Babel: An open chemical toolbox. J. Cheminf. 2011, "
        "3, 33. DOI: 10.1186/1758-2946-3-33"
    ),
    "mmgbsa": (
        "Genheden, S.; Ryde, U. The MM/PBSA and MM/GBSA methods to estimate "
        "ligand-binding affinities. Expert Opin. Drug Discovery 2015, 10 (5), "
        "449–461. DOI: 10.1517/17460441.2015.1032936"
    ),
    "amber": (
        "Case, D. A.; Babin, V.; Berryman, J. T.; Betz, R. M.; Cai, Q.; Cerutti, D. S.; "
        "Cheatham, T. E., III; Darden, T. A.; Duke, R. E.; Gohlke, H.; et al. "
        "AMBER 14; University of California: San Francisco, 2014."
    ),
}


# --------------------------------------------------------------------------- #
# Importación y registro de módulos científicos (Blueprints)
# --------------------------------------------------------------------------- #
# Cada módulo agrupa sus rutas (por ejemplo /api/dft, /api/docking, etc.).
# Se importa el blueprint y se registra en la app principal.

from modules.quimica_computacional import bp as quimica_bp
from modules.bioinformatica import bp as bioinformatica_bp
# from modules.hybrid_qmmm import bp as hybrid_qmmm_bp  # ELIMINADO (QM/MM)

app.register_blueprint(quimica_bp)
app.register_blueprint(bioinformatica_bp)
# app.register_blueprint(hybrid_qmmm_bp)                 # ELIMINADO (QM/MM)


# --------------------------------------------------------------------------- #
# Endpoint: Asistente "AI" de recomendación de parámetros
# --------------------------------------------------------------------------- #
@app.route("/api/ai_assistant", methods=["POST"])
def ai_assistant():
    """
    Recomienda parámetros de cálculo según la tarea solicitada y los
    recursos del sistema (RAM libre, número de CPUs).

    Entrada JSON (opcional):
        {
            "task": "dft" | "docking" | "md",
            "n_atoms": int
        }

    Salida JSON:
        {
            "suggestions": [str, ...],      # Mensajes legibles para el usuario
            "level": "ok" | "warning" | "danger",
            "recommended_functional": str,  # Funcional DFT sugerido
            "recommended_basis": str,       # Base DFT sugerida
            "recommended_cpu": int,         # Nº CPUs recomendado
            "ram_avail_gb": float,
            "cpu_count": int,
            "bse_available": bool,          # Backend SciPy/Engine disponible
            "openmmff_available": bool      # OpenMM/FF disponible
        }
    """
    data = request.json or {}
    task = data.get("task", "dft")
    n_atoms = int(data.get("n_atoms", 10))

    # Recursos actuales del sistema
    mem = psutil.virtual_memory()
    cpu_count = psutil.cpu_count()
    ram_gb = round(mem.available / (1024 ** 3), 2)

    # Valores por defecto de la recomendación
    sugg, level = [], "ok"
    rec_func, rec_basis, rec_cpu = "B3LYP", "def2-SVP", min(cpu_count, 4)

    # Reglas heurísticas según el tipo de tarea
    if task == "dft":
        if n_atoms > 80:
            sugg.append(" Sistema grande. Usa r²SCAN-3c.")
            level = "warning"
        if ram_gb < 2:
            level = "danger"

    elif task == "docking":
        if ram_gb < 2:
            level = "danger"
        if cpu_count >= 8:
            rec_cpu = min(cpu_count, 8)

    elif task == "md":
        if ram_gb < 4:
            level = "warning"

    # Si no hubo advertencias, indicamos que los recursos son óptimos.
    if not sugg:
        sugg.append(f"✓ Recursos óptimos ({ram_gb} GB libres).")

    return jsonify({
        "suggestions": sugg,
        "level": level,
        "recommended_functional": rec_func,
        "recommended_basis": rec_basis,
        "recommended_cpu": rec_cpu,
        "ram_avail_gb": ram_gb,
        "cpu_count": cpu_count,
        "bse_available": BSE_AVAILABLE,
        "openmmff_available": OPENMMFF_AVAILABLE,
    })


# --------------------------------------------------------------------------- #
# Endpoints básicos: raíz y ficheros de trabajos
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    """Sirve la página principal de la aplicación (index.html)."""
    return render_template("index.html")


@app.route("/jobs/<job_id>/<filename>")
def serve_job_file(job_id, filename):
    """
    Sirve ficheros generados por un trabajo (reportes, trayectorias, etc.).

    Se usa ``os.path.basename`` para evitar ataques de "path traversal"
    (por ejemplo peticiones con ``../``).
    """
    return send_from_directory(JOBS_DIR / job_id, os.path.basename(filename))


# --------------------------------------------------------------------------- #
# Endpoint: información del sistema
# --------------------------------------------------------------------------- #
@app.route("/api/system_info")
def system_info():
    """
    Devuelve información básica del sistema para el panel de estado:

        - cpu_count      : nº de núcleos lógicos.
        - cpu_percent    : uso de CPU instantáneo (no bloqueante).
        - ram_total_gb   : RAM total en GB.
        - ram_avail_gb   : RAM disponible en GB.
    """
    m = psutil.virtual_memory()
    return jsonify({
        "cpu_count": psutil.cpu_count(),
        "cpu_percent": psutil.cpu_percent(interval=None),
        "ram_total_gb": round(m.total / (1024 ** 3), 1),
        "ram_avail_gb": round(m.available / (1024 ** 3), 2),
    })


# --------------------------------------------------------------------------- #
# Endpoint: cancelar un trabajo
# --------------------------------------------------------------------------- #
@app.route("/api/cancelar/<job_id>", methods=["POST"])
def cancelar_job(job_id):
    """
    Marca el trabajo como cancelado y, si tiene un proceso externo
    asociado (``ACTIVE_PROCESSES``), intenta terminarlo.
    """
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({"error": "No encontrado"}), 404

        # Marcar el estado lógico del trabajo
        job["status"] = "cancelled"

        # Terminar el proceso subyacente (si existe)
        if job_id in ACTIVE_PROCESSES:
            try:
                ACTIVE_PROCESSES[job_id].kill()
            except Exception:
                # Si el proceso ya terminó o no se puede matar, lo ignoramos.
                pass

    return jsonify({"success": True})


# --------------------------------------------------------------------------- #
# Endpoint: consultar estado de un trabajo
# --------------------------------------------------------------------------- #
@app.route("/api/estado/<job_id>")
def obtener_estado(job_id):
    """
    Devuelve el diccionario completo del trabajo (progreso, resultados,
    estado, etc.) o 404 si no existe.
    """
    with JOBS_LOCK:
        job = JOBS.get(job_id)

    return jsonify(job) if job else (jsonify({"error": "No encontrado"}), 404)


# --------------------------------------------------------------------------- #
# Endpoint: generar reporte de un trabajo
# --------------------------------------------------------------------------- #
@app.route("/api/generate_report", methods=["POST"])
def generate_report():
    """
    Genera un informe en texto plano (``report.txt``) a partir de los
    datos almacenados en el trabajo indicado.

    Entrada JSON:
        { "job_id": "<id>" }

    Salida JSON en caso de éxito:
        {
            "success": true,
            "report_url": "/jobs/<job_id>/report.txt",
            "job_id": "<id>"
        }

    El formato del reporte depende del tipo de tarea:
        - DFT (por defecto)
        - MOLECULAR DOCKING
        - MOLECULAR DYNAMICS
        - STRUCTURE PREDICTION

    El reporte incluye una sección final "REFERENCES" con las citas
    bibliográficas de los paquetes de terceros efectivamente utilizados
    en el cálculo (ver diccionario global ``CITATIONS``).
    """
    try:
        data = request.json or {}
        job_id = data.get("job_id")

        # Recuperar el trabajo bajo lock
        with JOBS_LOCK:
            job = JOBS.get(job_id) if job_id else None

        if not job:
            return jsonify({"error": "Job no encontrado"}), 404

        # Asegurar carpeta del trabajo y ruta del reporte
        job_dir = JOBS_DIR / job_id
        job_dir.mkdir(exist_ok=True)
        report_path = job_dir / "report.txt"

        # Extraer campos relevantes
        task = str(job.get("task", ""))
        results = job.get("results", {}) or {}
        props = job.get("properties", {}) or {}
        vibs = job.get("vibraciones", []) or []

        # Determinar el tipo de reporte según el nombre de la tarea
        report_type = "DFT"
        if "docking" in task and "md" not in task:
            report_type = "MOLECULAR DOCKING"
        elif "md" in task:
            report_type = "MOLECULAR DYNAMICS"
        elif "alphafold" in task:
            report_type = "STRUCTURE PREDICTION"

        # ------------------------------------------------------------------ #
        # Utilidades de formato del reporte
        # ------------------------------------------------------------------ #
        W = 80
        def bar(c='='):
            """Devuelve una línea de ancho W con el carácter `c`."""
            return c * W

        def center(txt):
            """Centra el texto en una línea de ancho W."""
            return txt.center(W)

        # ------------------------------------------------------------------ #
        # Cabecera del reporte
        # ------------------------------------------------------------------ #
        out = [
            bar('*'),
            center("*" * 40),
            center("*        Q U A N T U M B I O   S T U D I O   P R O        *"),
            center("*" * 40),
            bar('*'),
            "",
            bar('='),
            center("COMPUTATIONAL CHEMISTRY REPORT"),
            bar('='),
            "",
            "",
            " JOB INFORMATION",
            " " + "-" * 15,
            f" Job ID             : {job_id}",
            f" Date               : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f" Calculation Type   : {report_type}",
            f" Status             : {job.get('status', 'N/A').upper()}",
            "",
        ]

        # ------------------------------------------------------------------ #
        # Parámetros del cálculo
        # ------------------------------------------------------------------ #
        out.append(" CALCULATION PARAMETERS")
        out.append(" " + "-" * 22)

        if report_type == "DFT":
            out.append(f" Method (Functional) : {job.get('functional', 'B3LYP')}")
            out.append(f" Basis Set           : {job.get('basis', 'def2-SVP')}")
            out.append(f" Task                : {job.get('dft_task', 'opt_freq')}")
            out.append(f" Engine              : {job.get('engine', 'orca').upper()}")
            if job.get("simulation_method"):
                out.append(f" Methodology         : {job.get('simulation_method')}")

        elif report_type == "MOLECULAR DYNAMICS":
            out.append(f" Methodology         : {job.get('simulation_method', 'Classic Molecular Dynamics (MM)')}")
            out.append(f" Force Field         : {job.get('forcefield', 'amber14-all.xml')}")
            out.append(f" Solvent Model       : {job.get('solvent', 'tip3p')}")
            out.append(f" Box Shape           : {job.get('box_shape', 'octahedron')}")
            out.append(f" CPU Cores Used      : {job.get('cpu_cores', 2)}")
            out.append(f" Total Steps         : {job.get('steps', 5000)}")
            out.append(f" Frames Saved        : {job.get('nframes', 10)}")
            # Si la simulación parte de un docking previo, se anotan
            # los datos de la pose seleccionada.
            if job.get("source_docking"):
                out.append(f" Source Docking      : {job.get('source_docking')}")
                out.append(f" Pose Selected       : {job.get('pose_index')}")
                out.append(f" Pose Score (Vina)   : {job.get('pose_score')} kcal/mol")

        elif report_type == "MOLECULAR DOCKING":
            out.append(f" Docking Software    : AutoDock Vina")
            out.append(f" Exhaustiveness      : {job.get('exhaustiveness', 8)}")
            out.append(f" CPU Cores Used      : {job.get('cpu_cores', 2)}")

        out.extend(["", ""])

        # ------------------------------------------------------------------ #
        # Resultados: DFT
        # ------------------------------------------------------------------ #
        if report_type == "DFT" and props:
            out.extend([" QUANTUM CHEMISTRY RESULTS", " " + "-" * 25])
            out.append(f" Total Energy        : {props.get('energy', 'N/A')} Ha")
            out.append(f" Convergence         : {props.get('converged', 'N/A')}")
            out.append(f" HOMO Energy         : {props.get('homo', 'N/A')} eV")
            out.append(f" LUMO Energy         : {props.get('lumo', 'N/A')} eV")
            out.append(f" HOMO-LUMO Gap       : {props.get('gap', 'N/A')} eV")
            out.append(f" Dipole Moment       : {props.get('dipole', 'N/A')} Debye")
            out.extend(["", ""])

        # ------------------------------------------------------------------ #
        # Resultados: Dinámica Molecular
        # ------------------------------------------------------------------ #
        if report_type == "MOLECULAR DYNAMICS" and "analytics" in results:
            a = results["analytics"]
            out.extend([" MD TRAJECTORY METRICS (Final Frame)", " " + "-" * 35])
            # Se toma el último valor de cada magnitud (frame final).
            out.append(f" Potential Energy    : {a['potential'][-1]:.2f} kcal/mol")
            out.append(f" Kinetic Energy      : {a['kinetic'][-1]:.2f} kcal/mol")
            out.append(f" Total Energy        : {a['total'][-1]:.2f} kcal/mol")
            out.append(f" System Temperature  : {a['temp'][-1]:.2f} K")
            out.append(f" RMSD (C-alpha)      : {a['rmsd'][-1]:.3f} Å")
            out.append(f" Radius of Gyration  : {a['rg'][-1]:.3f} Å")
            out.extend(["", ""])

            # Energía libre de unión estimada con MM/GBSA (si existe).
            mmgbsa = results.get("mmgbsa_binding_energy", "N/A")
            if mmgbsa and mmgbsa != "N/A":
                out.extend([" BINDING FREE ENERGY (MM/GBSA)", " " + "-" * 29])
                out.append(f" Delta G (Binding)   : {mmgbsa}")
                out.append(f" Methodology         : {results.get('mmgbsa_note', 'Amber14 / OBC2')}")
                out.extend(["", ""])

        # ------------------------------------------------------------------ #
        # Resultados: Docking molecular
        # ------------------------------------------------------------------ #
        if report_type == "MOLECULAR DOCKING" and results:
            out.extend([
                " MOLECULAR DOCKING RESULTS",
                " " + "-" * 25,
                f" Best Score          : {results.get('score')}",
                "",
                " " + "-" * 62,
                "   POSE        SCORE (kcal/mol)        Ki (M)",
                " " + "-" * 62,
            ])
            # Listado de puntuaciones por pose y su Ki asociado (si existe).
            for i, s in enumerate(results.get("scores", [])):
                ki_val = (
                    results.get('ki_values', [0] * 10)[i]
                    if i < len(results.get('ki_values', []))
                    else 0
                )
                out.append(f"   {i+1:<8d}     {s:14.4f}          {ki_val:.4e}")
            out.extend([" " + "-" * 62, "", ""])

        # ------------------------------------------------------------------ #
        # Resultados: frecuencias vibracionales (DFT)
        # ------------------------------------------------------------------ #
        if report_type == "DFT" and vibs:
            out.extend([" VIBRATIONAL FREQUENCIES (cm-1)", " " + "-" * 30])
            # Se imprimen 4 frecuencias por línea.
            for i in range(0, len(vibs), 4):
                chunk = vibs[i:i + 4]
                out.append("   " + "".join([f"{v['frecuencia']:>10.2f}" for v in chunk]))
            out.extend(["", ""])

        # ------------------------------------------------------------------ #
        # Referencias bibliográficas (citas de software)                     #
        # ------------------------------------------------------------------ #
        # Se construye el conjunto de citas pertinentes según el tipo de
        # cálculo y las herramientas realmente utilizadas. Solo se incluyen
        # las citas de los paquetes que han intervenido, evitando ruido.
        refs = set()

        if report_type == "DFT":
            engine = str(job.get("engine", "orca")).lower()
            if engine == "pyscf":
                refs.add("pyscf")
            elif engine == "orca":
                refs.add("orca")
            # BSE solo si estaba disponible y se usó para obtener la base
            if BSE_AVAILABLE:
                refs.add("bse")
            # RDKit se usa en la generación/parseo de estructuras
            refs.add("rdkit")

        elif report_type == "MOLECULAR DOCKING":
            # Vina + OpenBabel (preparación PDBQT) + RDKit (parseo SMILES)
            refs.update(["vina", "openbabel", "rdkit"])

        elif report_type == "MOLECULAR DYNAMICS":
            # OpenMM como motor; OpenFF para el ligando; PDBFixer para
            # reparar la proteína; AMBER para los parámetros; MM/GBSA
            # para la afinidad; OpenBabel y RDKit en el pipeline.
            refs.update([
                "openmm", "openff", "pdbfixer", "amber",
                "mmgbsa", "openbabel", "rdkit",
            ])
            # Si la MD partió de un docking, se cita también Vina.
            if job.get("source_docking"):
                refs.add("vina")

        elif report_type == "STRUCTURE PREDICTION":
            refs.add("alphafold")
            # Si el mensaje final indica que se usó ESMFold (fallback),
            # añadimos su cita también.
            if "ESMFold" in str(job.get("message", "")):
                refs.add("esmfold")

        # Opcional: si en el futuro guardas un flag ``used_pymol`` en el job,
        # puedes activar esta cita:
        # if job.get("used_pymol"):
        #     refs.add("pymol")

        if refs:
            out.extend([
                "",
                " REFERENCES",
                " " + "-" * 12,
            ])
            for key in sorted(refs):
                if key in CITATIONS:
                    out.append(f" - {CITATIONS[key]}")
            out.extend(["", ""])

        # ------------------------------------------------------------------ #
        # Pie del reporte
        # ------------------------------------------------------------------ #
        out.extend([
            "", "",
            bar(),
            center("*" * 40),
            center("*          JOB COMPLETED NORMALLY       *"),
            center("*" * 40),
            bar(),
            "",
            center("Generated by QuantumBio Studio Pro"),
            center(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            "",
        ])

        # Escribir el reporte en disco (UTF-8)
        report_path.write_text("\n".join(out), encoding="utf-8")

        return jsonify({
            "success": True,
            "report_url": f"/jobs/{job_id}/report.txt",
            "job_id": job_id,
        })

    except Exception as e:
        logger.exception("Error generando reporte")
        return jsonify({"error": f"Error generando reporte: {e}"}), 500


# --------------------------------------------------------------------------- #
# Manejadores de errores personalizados
# --------------------------------------------------------------------------- #
@app.errorhandler(404)
def _not_found(e):
    """
    Para rutas /api/* devolvemos JSON; para el resto, dejamos que Flask
    use su página 404 por defecto.
    """
    if request.path.startswith("/api/"):
        return jsonify({"error": f"Endpoint no encontrado: {request.path}"}), 404
    return e


@app.errorhandler(500)
def _server_error(e):
    """
    Igual que 404: JSON para /api/*, página por defecto para el resto.
    """
    if request.path.startswith("/api/"):
        return jsonify({"error": "Error interno del servidor"}), 500
    return e


# --------------------------------------------------------------------------- #
# Hook global: cabeceras de caché
# --------------------------------------------------------------------------- #
@app.after_request
def add_header(response):
    """
    Desactiva el cacheo de respuestas para que los clientes siempre
    reciban el estado más reciente (resultados, trayectorias, etc.).
    """
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response


# --------------------------------------------------------------------------- #
# Arranque del servidor
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # ``debug=False`` y ``use_reloader=False`` son importantes porque
    # los trabajos usan procesos/subprocesos que no deben duplicarse
    # por el auto-reloader de Flask.
    app.run(
        debug=False,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5001")),
        use_reloader=False,
    )
