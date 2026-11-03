"""
Módulo de Bioinformática de "QuantumBio Studio Pro".

Este módulo registra un Blueprint de Flask (``bioinformatica``) que expone
la API y los motores de:

    - **Preparación de estructuras** (PyMOL vía XML-RPC).
    - **Predicción de estructura** (AlphaFold DB + ESMFold API).
    - **Docking molecular** (AutoDock Vina + OpenBabel).
    - **Dinámica Molecular** (OpenMM + OpenFF + PDBFixer + MM/GBSA).
    - **Exportación a PyMOL** (comandos vía XML-RPC).

Flujo típico desde el cliente web
---------------------------------
    1. ``/api/prepare_pdb``         — Limpieza de PDB con PyMOL.
    2. ``/api/fetch_pdb/<id>``      — Descarga de RCSB PDB.
    3. ``/api/alphafold``           — Plegamiento por secuencia.
    4. ``/api/docking``             — Docking molecular (Vina).
    5. ``/api/md``                  — MD proteína-ligando (7 fases + MM/GBSA).
    6. ``/api/md_from_docking``     — MD partiendo de una pose de docking.
    7. ``/api/export_to_pymol``     — Exportar resultados a PDB para PyMOL.
    8. ``/api/pymol_command``       — Enviar comandos a PyMOL (XML-RPC).

Convenciones
------------
    - Igual que en ``quimica_computacional``, los nombres de ``core`` se
      importan con ``globals().update(...)`` para no prefijarlos con ``_core.``.
    - Los trabajos se ejecutan en hilos y actualizan ``JOBS[job_id]`` vía
      ``_update_job``.
    - Los subprocesos externos (ORCA, Vina, obabel, PyMOL) se invocan con
      ``subprocess`` y los que admiten cancelación se registran en
      ``ACTIVE_PROCESSES``.
"""

# Vuelca los nombres públicos de "core" al espacio global de este módulo.
# Así podemos usar directamente JOBS, JOBS_DIR, JOBS_LOCK, ACTIVE_PROCESSES,
# logger, omm_app, mm, unit, SystemGenerator, PDBFIXER_AVAILABLE,
# OPENMMFF_AVAILABLE, BIO_AVAILABLE, PDBParser, OFFMolecule, Chem,
# rdDetermineBonds, requests, xmlrpc, subprocess, threading, uuid, time,
# np, math, re, StringIO, _update_job, check_job_cancelled, JobCancelled,
# _split_models_from_file, _build_vibrations, calc_rmsd, calc_rg,
# calcular_ki, etc., sin necesidad de prefijarlos.
import core as _core
globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

from flask import Blueprint

# Blueprint que agrupa todas las rutas de este módulo.
bp = Blueprint("bioinformatica", __name__)


# --------------------------------------------------------------------------- #
# Endpoint: preparación de PDB con PyMOL (vía XML-RPC)
# --------------------------------------------------------------------------- #
@bp.route("/api/prepare_pdb", methods=["POST"])
def prepare_pdb():
    """
    Limpia y prepara una estructura PDB usando una instancia de PyMOL
    corriendo en modo servidor XML-RPC (``pymol -R`` en el puerto 9123).

    Pasos ejecutados en PyMOL:
        1. ``delete all``   — limpiar sesión.
        2. ``load <input>`` — cargar el PDB del usuario.
        3. ``remove solvent`` / ``remove inorganic`` — quitar disolvente e iones.
        4. ``h_add``        — añadir hidrógenos.
        5. ``save <output>``— guardar PDB limpio.

    Entrada JSON:
        { "pdb_text": "<contenido PDB>" }

    Salida JSON:
        - Éxito: { "success": true, "pdb_text": "<PDB limpio>" }
        - Error: { "error": "<mensaje>" } con código 400/500.
    """
    data = request.json or {}
    pdb_text = data.get("pdb_text", "")
    if not pdb_text:
        return jsonify({"error": "No hay estructura PDB"}), 400

    try:
        # Crear carpeta del trabajo y ficheros de entrada/salida
        job_id = str(uuid.uuid4())
        job_dir = JOBS_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        input_file = job_dir / "input.pdb"
        output_file = job_dir / "cleaned.pdb"
        input_file.write_text(pdb_text)

        # Conectar con PyMOL (servidor XML-RPC en localhost:9123)
        server = xmlrpc.client.ServerProxy('http://127.0.0.1:9123', allow_none=True)
        server.do("delete all")
        server.do(f"load {input_file.resolve()}, prot")
        server.do("remove solvent")
        server.do("remove inorganic")
        server.do("h_add")
        server.do(f"save {output_file.resolve()}")
        server.do("delete all")

        if output_file.exists():
            return jsonify({"success": True, "pdb_text": output_file.read_text()})
        return jsonify({"error": "PyMOL no pudo guardar el archivo"}), 500

    except ConnectionRefusedError:
        return jsonify({"error": "PyMOL no conectado. Ejecuta: pymol -R"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --------------------------------------------------------------------------- #
# Motor: predicción de estructura (AlphaFold DB + ESMFold API)
# --------------------------------------------------------------------------- #
def run_alphafold_job(job_id, sequence):
    """
    Predice (o recupera) una estructura proteica a partir de su secuencia.

    Estrategia:
        1. Buscar la secuencia en UniProt (por secuencia exacta).
        2. Si se encuentra, descargar el modelo de AlphaFold DB (AF-<id>-F1).
        3. Si no se encuentra, invocar la API pública de ESMFold.
           - Límite: 400 aminoácidos.
           - Reintentos con backoff exponencial ante 429/5xx/timeouts.

    Parámetros:
        job_id   : str, ID del trabajo.
        sequence : str, secuencia de aminoácidos (se limpia a solo letras).

    Deja el resultado en ``JOBS[job_id]["results"]["pdb_result"]``.
    """
    try:
        check_job_cancelled(job_id)

        # Limpiar secuencia: solo letras, mayúsculas
        seq_clean = re.sub(r'[^A-Z]', '', sequence.upper())
        if not seq_clean:
            raise ValueError("La secuencia está vacía.")

        # ---------- Intento 1: UniProt + AlphaFold DB -------------------- #
        _update_job(job_id, progress=10, message="Buscando secuencia en UniProt...")
        try:
            url = "https://rest.uniprot.org/uniprotkb/search"
            params = {"query": f"sequence:{seq_clean}", "format": "json", "size": 1}
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            results = r.json().get("results", [])

            if results:
                uniprot_id = results[0]["primaryAccession"]
                _update_job(job_id, progress=40,
                            message=f"ID {uniprot_id} encontrado. Descargando de AlphaFold DB...")

                af_url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
                af_r = requests.get(af_url, timeout=20)
                if af_r.status_code == 200:
                    _update_job(
                        job_id, status="done", progress=100,
                        message=f"Estructura obtenida de AlphaFold DB ({uniprot_id})",
                        results={"pdb_result": af_r.text},
                    )
                    return
        except Exception as e:
            # Si UniProt/AlphaFold falla, se continúa con ESMFold
            logger.warning(f"Fallo en la búsqueda UniProt: {e}")

        # ---------- Intento 2: ESMFold API ------------------------------- #
        _update_job(job_id, progress=50,
                    message="No encontrado en BD. Prediciendo con ESMFold API...")

        if len(seq_clean) > 400:
            raise ValueError("La secuencia es muy larga para la API de ESMFold (máx 400).")

        url = 'https://api.esmatlas.com/foldSequence/v1/pdb/'
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        max_retries = 3
        last_error = None

        for attempt in range(1, max_retries + 1):
            check_job_cancelled(job_id)
            try:
                # Timeouts (conexión 10s, lectura 300s)
                response = requests.post(
                    url, headers=headers, data=seq_clean,
                    timeout=(10, 300), verify=False,
                )
                if response.status_code == 200:
                    _update_job(
                        job_id, status="done", progress=100,
                        message="Plegamiento completado (ESMFold API)",
                        results={"pdb_result": response.text},
                    )
                    return

                last_error = f"HTTP {response.status_code}"
                # Errores transitorios: reintentar con backoff exponencial
                if response.status_code in (429, 500, 502, 503, 504):
                    time.sleep(2 ** attempt)
                    _update_job(job_id, progress=50 + attempt * 10,
                                message=f"Servidor saturado ({last_error}). Reintento {attempt}...")
                    continue
                break  # error no recuperable

            except requests.exceptions.Timeout:
                last_error = "Timeout"
                time.sleep(2 ** attempt)
                _update_job(job_id, progress=50 + attempt * 10,
                            message=f"Tiempo de espera agotado. Reintento {attempt}...")

        raise RuntimeError(f"ESMFold no respondió. Último error: {last_error}")

    except JobCancelled:
        _update_job(job_id, status="cancelled", message="Cancelado")
    except Exception as e:
        _update_job(job_id, status="error", progress=100, message=f"Error: {str(e)}")


# --------------------------------------------------------------------------- #
# Helper: obtención del "record type" de una línea PDB
# --------------------------------------------------------------------------- #
def _pdb_record(line):
    """
    Devuelve las 6 primeras columnas de una línea PDB (tipo de registro:
    ``ATOM``, ``HETATM``, ``TER``, ``END``, etc.).
    """
    return line[:6].strip() if len(line) >= 6 else ""


# ==============================================================================
# HELPERS PARA MM/GBSA Y MANIPULACIÓN DE TOPOLOGÍAS
# ==============================================================================
def _extract_sub_topology(topology, atom_indices_set):
    """
    Crea una nueva ``Topology`` de OpenMM que solo contiene los átomos
    indicados en ``atom_indices_set``, preservando cadenas, residuos,
    átomos y enlaces.

    Esto es necesario para MM/GBSA: necesitamos la topología del complejo,
    de la proteína sola y del ligando solo, para calcular por separado sus
    energías y obtener ΔG = E_complejo - E_proteína - E_ligando.

    Retorna:
        (new_top, atom_map):
            - new_top  : nueva Topology de OpenMM.
            - atom_map : dict {índice_original: átomo_nuevo} (útil para
                         depurar y para reconstruir posiciones).
    """
    new_top = omm_app.Topology()
    atom_map = {}

    for chain in topology.chains():
        new_chain = None
        for residue in chain.residues():
            res_atoms = [a for a in residue.atoms() if a.index in atom_indices_set]
            if not res_atoms:
                continue

            if new_chain is None:
                new_chain = new_top.addChain(chain.id)

            # Añadir residuo (compatible con distintas versiones de OpenMM)
            try:
                new_res = new_top.addResidue(
                    residue.name, new_chain, residue.id,
                    residue.insertionCode if residue.insertionCode else '',
                )
            except TypeError:
                new_res = new_top.addResidue(residue.name, new_chain, str(residue.id))

            # Añadir átomos
            for atom in res_atoms:
                try:
                    new_atom = new_top.addAtom(atom.name, atom.element, new_res, atom.id)
                except TypeError:
                    new_atom = new_top.addAtom(atom.name, atom.element, new_res)
                atom_map[atom.index] = new_atom

    # Restaurar los enlaces para que OpenMM reconozca los aminoácidos
    for bond in topology.bonds():
        if bond[0].index in atom_map and bond[1].index in atom_map:
            new_top.addBond(atom_map[bond[0].index], atom_map[bond[1].index])

    return new_top, atom_map


def _single_point_energy_mmgbsa(topology, positions_sub_nm, sysgen_gb, platform, ligand_mols):
    """
    Energía de punto simple (con minimización corta) usando el sistema
    GB/OBC2 para MM/GBSA. Devuelve la energía potencial en kcal/mol.

    Parámetros:
        topology          : Topology del subsistema (complejo, proteína o ligando).
        positions_sub_nm  : posiciones (en nm) correspondientes a ese subsistema.
        sysgen_gb         : SystemGenerator configurado con GB/OBC2.
        platform          : plataforma de OpenMM (CPU).
        ligand_mols       : lista de moléculas OpenFF (para OpenFFTemplateGenerator).
    """
    system = sysgen_gb.create_system(topology, molecules=ligand_mols)
    integrator = mm.VerletIntegrator(0.001)
    context = mm.Context(system, integrator, platform)
    try:
        context.setPositions(positions_sub_nm)
        # Minimización ligera para evitar choques antes de medir energía
        mm.LocalEnergyMinimizer.minimize(context, tolerance=10.0, maxIterations=100)
        state = context.getState(getEnergy=True)
        return state.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
    finally:
        # Liberar recursos de OpenMM
        del context, integrator


def _compute_mmgbsa(topology, frames_positions_nm, protein_indices, ligand_indices,
                    sysgen_gb, platform, ligand_mols, n_samples=5):
    """
    Calcula MM/GBSA sobre los últimos ``n_samples`` frames de la trayectoria.

    Para cada frame:
        ΔG = E(complejo) - E(proteína) - E(ligando)

    Retorna:
        (mean_dG, nota):
            - mean_dG : energía media de unión en kcal/mol (o None si falla).
            - nota    : string descriptivo (frames usados, media ± desviación).
    """
    if not frames_positions_nm:
        return None, "sin frames de producción"

    n_use = min(n_samples, len(frames_positions_nm))
    complex_indices = list(protein_indices) + list(ligand_indices)
    protein_idx_arr = np.array(protein_indices)
    ligand_idx_arr = np.array(ligand_indices)
    complex_idx_arr = np.array(complex_indices)

    # Extraer sub-topologías para complejo, proteína y ligando
    try:
        complex_top, _ = _extract_sub_topology(topology, set(complex_indices))
        protein_top, _ = _extract_sub_topology(topology, set(protein_indices))
        ligand_top, _ = _extract_sub_topology(topology, set(ligand_indices))
    except Exception as e:
        return None, f"error construyendo topologías: {e}"

    dGs = []
    for j, pos_nm in enumerate(frames_positions_nm[-n_use:]):
        try:
            e_complex = _single_point_energy_mmgbsa(
                complex_top, pos_nm[complex_idx_arr], sysgen_gb, platform, ligand_mols)
            e_protein = _single_point_energy_mmgbsa(
                protein_top, pos_nm[protein_idx_arr], sysgen_gb, platform, ligand_mols)
            e_ligand = _single_point_energy_mmgbsa(
                ligand_top, pos_nm[ligand_idx_arr], sysgen_gb, platform, ligand_mols)

            dG = e_complex - e_protein - e_ligand
            dGs.append(dG)
            logger.info(
                f"[MM/GBSA] frame {j+1}/{n_use}: E_complex={e_complex:.2f} | "
                f"E_protein={e_protein:.2f} | E_ligand={e_ligand:.2f} | "
                f"ΔG={dG:.2f} kcal/mol"
            )
        except Exception as e:
            logger.warning(f"[MM/GBSA] frame {j+1} falló: {e}")
            continue

    if not dGs:
        return None, "todos los frames fallaron"

    mean_dG = float(np.mean(dGs))
    std_dG = float(np.std(dGs))
    return mean_dG, f"{len(dGs)} frames, ΔG = {mean_dG:.2f} ± {std_dG:.2f} kcal/mol"


# ==============================================================================
# MOTOR PRINCIPAL: MD PROTEÍNA-LIGANDO (7 FASES + MM/GBSA + RENDER PIPELINE)
# ==============================================================================
def run_md_job(job_id, pdb_text, ligand_pdb=None, steps=5000, nframes=10,
               solvent_model="tip3p", forcefield="amber14-all.xml",
               cpu_cores=2, box_shape="octahedron"):
    """
    Ejecuta una simulación de Dinámica Molecular proteína-ligando con OpenMM.

    Fases del protocolo
    -------------------
        Fase 0  Analizar componentes (proteína + ligando HETATM).
        Fase 1  PDBFixer: añadir átomos/hidrógenos faltantes a pH 7.0.
        Fase 2  Parametrizar ligando con OpenFF-2.1.0 (vía OpenBabel + RDKit).
        Fase 3  Solvatar (caja + 0.15 M NaCl, neutralizar).
        Fase 4  Minimización (tol=10 kJ/mol·nm, maxIter=10000).
        Fase 5  NVT: calentamiento 0→300 K con restricciones posicionales.
        Fase 6  NPT: barostato MonteCarlo 1 bar / 300 K.
        Fase 7  Producción: trayectoria de ``nframes`` frames.
        Post    MM/GBSA (OBC2) sobre los últimos frames.

    Además, construye un ``gl_render_pipeline`` (dict) que indica al
    frontend cómo renderizar la trayectoria (estilos, cámara, filtros).

    Parámetros:
        job_id        : str
        pdb_text      : str, PDB de la proteína.
        ligand_pdb    : str|None, PDB del ligando (opcional).
        steps         : int, pasos totales de producción (mín. nframes*50).
        nframes       : int, nº de frames a guardar (2..200).
        solvent_model : str, modelo de agua ("tip3p", "spce", ...) o "none".
        forcefield    : str, fichero de campo de fuerzas de proteína.
        cpu_cores     : int, nº de hilos CPU.
        box_shape     : str, forma de la caja ("octahedron", "cube", ...).
    """
    try:
        check_job_cancelled(job_id)

        # Verificación de dependencias
        if not OPENMMFF_AVAILABLE:
            _update_job(job_id, status="error", progress=100,
                        message="openmmforcefields no instalado.")
            return

        # -------- Metadatos del job (para job MD lanzado desde docking) --- #
        with JOBS_LOCK:
            job_meta = dict(JOBS.get(job_id, {}))
        source_docking = job_meta.get("source_docking")
        pose_index = job_meta.get("pose_index")
        pose_score = job_meta.get("pose_score")
        ligand_atoms_meta = job_meta.get("ligand_atoms")

        # Normalización de parámetros
        try:
            nf = max(2, min(int(nframes), 200))
        except Exception:
            nf = 10
        try:
            steps = max(nf * 50, int(steps))
        except Exception:
            steps = 5000

        wd = JOBS_DIR / job_id
        wd.mkdir(exist_ok=True)
        original = wd / "original.pdb"

        # -------- Ensamblado del PDB inicial (proteína + ligando) -------- #
        protein_block = "\n".join(
            line for line in (pdb_text or "").splitlines()
            if _pdb_record(line) in ("ATOM", "HETATM", "TER")
        )
        ligand_block = "\n".join(
            line for line in (ligand_pdb or "").splitlines()
            if _pdb_record(line) in ("ATOM", "HETATM")
        )
        if not protein_block.strip():
            raise ValueError("La estructura de la proteína no contiene átomos PDB válidos.")

        combined = protein_block.rstrip()
        if ligand_block.strip():
            combined += "\nTER\n" + ligand_block.rstrip()
        original.write_text(combined + "\nEND\n")

        has_ligand = bool(ligand_block.strip())
        n_lig_actual = sum(1 for l in ligand_block.splitlines() if l.startswith("HETATM")) if has_ligand else 0
        n_prot_actual = sum(1 for l in protein_block.splitlines() if l.startswith(("ATOM  ", "HETATM")))

        msg_lig = (
            f"(proteína: {n_prot_actual} át, ligando: {n_lig_actual} át ✓)"
            if has_ligand else f"SOLO proteína ({n_prot_actual} át)"
        )
        _update_job(job_id, status="running", progress=5,
                    message=f"[Fase 0] Analizando componentes... {msg_lig}")

        # -------- Detección de HETATM candidatos a ligando ---------------- #
        IGNORE_RESIDUES = {
            'HOH', 'WAT', 'OH', 'OH2', 'H2O', 'O', 'DOD', 'TIP3', 'SOL',
            'NA', 'CL', 'K', 'MG', 'ZN', 'CA', 'MN', 'FE', 'CU', 'CO',
            'NI', 'CD', 'HG', 'HEM', 'FAD', 'NAD', 'ATP', 'ADP', 'AMP',
            'GDP', 'GTP', 'EOH',
        }
        original_content = original.read_text()
        hetatm_by_res = {}
        for line in original_content.split('\n'):
            if line.startswith('HETATM'):
                res_name = line[17:20].strip()
                if res_name in IGNORE_RESIDUES:
                    continue
                if res_name not in hetatm_by_res:
                    hetatm_by_res[res_name] = []
                hetatm_by_res[res_name].append(line)

        # -------- FASE 2: Parametrización del ligando -------------------- #
        _update_job(job_id, progress=10,
                    message="[Fase 2] Parametrizando ligando con OpenFF-2.1.0...")

        valid_res_names = []
        ligand_mols = []
        lig_top = None
        lig_pos = None

        for res_name, lines in hetatm_by_res.items():
            temp_pdb = wd / f"temp_{res_name}.pdb"
            temp_pdb_h = wd / f"temp_{res_name}_h.pdb"

            with open(temp_pdb, 'w') as f:
                f.write('\n'.join(lines) + '\nEND\n')

            try:
                # 1. Añadir hidrógenos con OpenBabel
                subprocess.run(
                    ["obabel", str(temp_pdb), "-O", str(temp_pdb_h), "-h"],
                    check=True, capture_output=True,
                )

                # 2. Leer PDB hidrogenado con RDKit
                mol = Chem.MolFromPDBFile(str(temp_pdb_h), removeHs=False, sanitize=False)
                if mol is None:
                    raise ValueError("RDKit no pudo parsear el archivo PDB.")

                # 3. Determinar enlaces y sanitizar
                rdDetermineBonds.DetermineBonds(mol, charge=0)
                Chem.SanitizeMol(mol)

                # 4. Convertir a OpenFF y asignar nombre
                off_mol = OFFMolecule.from_rdkit(mol)
                off_mol.name = res_name
                ligand_mols.append(off_mol)
                valid_res_names.append(res_name)

                # 5. Generar Topology y posiciones de OpenMM desde OpenFF
                lig_top = off_mol.to_topology().to_openmm()
                lig_pos = off_mol.conformers[0].to('nanometer').magnitude * unit.nanometer

                for res in lig_top.residues():
                    res.name = res_name

                logger.info(f"✓ Ligando parametrizado correctamente: {res_name}")

            except Exception as e:
                # Un ligando problemático no debe romper todo el job
                logger.warning(f"Ignorando {res_name}: {e}")

        if valid_res_names:
            _update_job(job_id, progress=15,
                        message=f"[Fase 2] Ligandos parametrizados: {', '.join(valid_res_names)}")
        elif has_ligand:
            _update_job(job_id, progress=15,
                        message="Ligando no parametrizable — MD solo proteína")

        # -------- SystemGenerator (campo de fuerzas + OpenFF) ------------ #
        system_generator = SystemGenerator(
            forcefields=[forcefield, 'amber14/tip3pfb.xml'],
            small_molecule_forcefield='openff-2.1.0',
            forcefield_kwargs={'constraints': omm_app.HBonds, 'rigidWater': True},
        )
        if ligand_mols:
            system_generator.add_molecules(ligand_mols)

        # Filtrar HETATM no parametrizables del PDB de entrada
        new_lines = []
        for line in original_content.split('\n'):
            if line.startswith('HETATM'):
                res_name = line[17:20].strip()
                if res_name in IGNORE_RESIDUES or res_name not in valid_res_names:
                    continue
            new_lines.append(line)
        filtered_pdb = wd / "filtered_input.pdb"
        filtered_pdb.write_text('\n'.join(new_lines))

        # -------- FASE 1: PDBFixer (solo proteína) ------------------------ #
        if not PDBFIXER_AVAILABLE:
            raise RuntimeError("PDBFixer no instalado.")
        _update_job(job_id, progress=20,
                    message="[Fase 1] PDBFixer: átomos/residuos faltantes + H a pH 7.0...")

        protein_lines = [l for l in original_content.split('\n') if l.startswith(('ATOM', 'TER'))]
        protein_only_pdb = wd / "protein_only.pdb"
        protein_only_pdb.write_text('\n'.join(protein_lines) + '\nEND\n')

        fixer = PDBFixer(filename=str(protein_only_pdb))
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(pH=7.0)
        fixed_pdb = wd / "fixed_protein.pdb"
        with open(fixed_pdb, "w") as f:
            omm_app.PDBFile.writeFile(fixer.topology, fixer.positions, f)

        pdb = omm_app.PDBFile(str(fixed_pdb))
        modeller = omm_app.Modeller(pdb.topology, pdb.positions)

        # -------- FASE 3: Solvatación ------------------------------------ #
        solvent_added = solvent_model in ("tip3p", "spce", "tip4pew", "tip4pfb")
        if solvent_added:
            # El ligando se añade ANTES de la solvatación para evitar solapamientos
            if lig_top is not None and lig_pos is not None:
                modeller.add(lig_top, lig_pos)
                logger.info("✓ Ligando añadido antes de la solvatación.")

            _update_job(job_id, progress=30,
                        message=f"[Fase 3] Solvatando ({solvent_model}, caja {box_shape}, "
                                f"padding 1.0 nm, 0.15 M NaCl)...")

            modeller.addSolvent(
                system_generator.forcefield,
                model=solvent_model,
                boxShape=box_shape,
                padding=1.0 * unit.nanometer,
                neutralize=True,
                ionicStrength=0.15 * unit.molar,
            )
        else:
            _update_job(job_id, progress=30,
                        message="Modo sin solvente explícito — NPT omitido")

        # -------- Construcción del sistema de equilibrio ----------------- #
        _update_job(job_id, progress=35, message="Creando sistema molecular...")
        system = system_generator.create_system(modeller.topology, molecules=ligand_mols)
        integrator = mm.LangevinIntegrator(300 * unit.kelvin,
                                           1 / unit.picosecond,
                                           0.002 * unit.picoseconds)

        platform = mm.Platform.getPlatformByName('CPU')
        properties = {'Threads': str(cpu_cores)}
        simulation = omm_app.Simulation(modeller.topology, system, integrator, platform, properties)
        simulation.context.setPositions(modeller.positions)
        simulation.context.setVelocitiesToTemperature(300 * unit.kelvin)

        topology = modeller.topology

        # -------- Identificación de átomos de interés -------------------- #
        AMINO_ACIDS = {
            'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
            'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
            'HID', 'HIE', 'HIP', 'HSD', 'HSE', 'HSP', 'CYX', 'CYM', 'ASH', 'GLH',
            'LYN', 'ARN', 'SEP', 'TPO', 'PTR', 'ACE', 'NME', 'NHE',
        }
        solute_resnames = set(valid_res_names)

        protein_indices = [a.index for a in topology.atoms() if a.residue.name in AMINO_ACIDS]
        ligand_indices = [a.index for a in topology.atoms() if a.residue.name in solute_resnames]

        solute_atoms = [
            a for a in topology.atoms()
            if a.residue.name in AMINO_ACIDS or a.residue.name in solute_resnames
        ]
        solute_indices = [a.index for a in solute_atoms]
        solute_heavy_indices = [a.index for a in solute_atoms if a.element.symbol != 'H']

        # Carbonos alfa (para RMSD)
        ca_indices = [
            a.index for a in topology.atoms()
            if a.name == 'CA' and a.residue.name in AMINO_ACIDS
        ]
        if not ca_indices:
            # Fallback: usar todos los átomos de proteína
            ca_indices = list(protein_indices) if protein_indices else list(range(topology.getNumAtoms()))
        ca_indices = np.array(ca_indices)

        # -------- FASE 4: Minimización ----------------------------------- #
        _update_job(job_id, progress=40,
                    message=f"[Fase 4] Minimizando ({system.getNumParticles()} partículas, "
                            f"tol=10 kJ/(mol·nm), maxIter=10000)...")
        simulation.minimizeEnergy(
            tolerance=10.0 * unit.kilojoule_per_mole / unit.nanometer,
            maxIterations=10000,
        )
        check_job_cancelled(job_id)

        # -------- FASES 5 y 6: NVT + NPT --------------------------------- #
        if solvent_added:
            # Restricción armónica sobre átomos pesados del soluto
            restraint = mm.CustomExternalForce(
                '0.5 * k_restraint * ((x - x0)^2 + (y - y0)^2 + (z - z0)^2)'
            )
            restraint.setName('RestraintForce')
            restraint.addGlobalParameter('k_restraint', 500.0)
            restraint.addPerParticleParameter('x0')
            restraint.addPerParticleParameter('y0')
            restraint.addPerParticleParameter('z0')

            state0 = simulation.context.getState(getPositions=True)
            pos0_nm = state0.getPositions(asNumpy=True)
            for idx in solute_heavy_indices:
                p = pos0_nm[idx].value_in_unit(unit.nanometer)
                restraint.addParticle(idx, [float(p[0]), float(p[1]), float(p[2])])

            system.addForce(restraint)
            simulation.context.reinitialize(preserveState=True)

            # -------- NVT: calentamiento gradual 0→300 K ---------------- #
            _update_job(job_id, progress=45,
                        message="[Fase 5] NVT: calentamiento 0→300 K con restricciones...")
            nvt_steps = max(5000, steps // 4)
            nvt_chunks = 10
            spc_nvt = max(1, nvt_steps // nvt_chunks)
            for c in range(nvt_chunks):
                check_job_cancelled(job_id)
                T = 300.0 * (c + 1) / nvt_chunks
                integrator.setTemperature(T * unit.kelvin)
                simulation.step(spc_nvt)
                _update_job(job_id,
                            progress=45 + int(((c + 1) / nvt_chunks) * 10),
                            message=f"[Fase 5] NVT {T:.0f} K ({c + 1}/{nvt_chunks})")

            # -------- NPT: barostato 1 bar / 300 K ---------------------- #
            _update_job(job_id, progress=55,
                        message="[Fase 6] NPT: barostato 1 bar / 300 K...")
            barostat = mm.MonteCarloBarostat(1.0 * unit.bar, 300 * unit.kelvin, 25)
            system.addForce(barostat)
            simulation.context.reinitialize(preserveState=True)
            integrator.setTemperature(300.0 * unit.kelvin)

            npt_steps = max(5000, steps // 4)
            npt_chunks = 5
            spc_npt = max(1, npt_steps // npt_chunks)
            for c in range(npt_chunks):
                check_job_cancelled(job_id)
                simulation.step(spc_npt)
                _update_job(job_id,
                            progress=55 + int(((c + 1) / npt_chunks) * 10),
                            message=f"[Fase 6] NPT {c + 1}/{npt_chunks}")

            # -------- FASE 7: producción sin restricciones -------------- #
            _update_job(job_id, progress=65,
                        message="[Fase 7] Producción: removiendo restricciones...")
            state_eq = simulation.context.getState(
                getPositions=True, getVelocities=True, getEnergy=True,
                enforcePeriodicBox=True,
            )

            # Reconstruir sistema SIN restricción + con barostato
            system_prod = system_generator.create_system(topology, molecules=ligand_mols)
            barostat_prod = mm.MonteCarloBarostat(1.0 * unit.bar, 300 * unit.kelvin, 25)
            system_prod.addForce(barostat_prod)
            integrator_prod = mm.LangevinIntegrator(300 * unit.kelvin,
                                                    1 / unit.picosecond,
                                                    0.002 * unit.picoseconds)
            simulation = omm_app.Simulation(topology, system_prod, integrator_prod,
                                            platform, properties)
            simulation.context.setState(state_eq)
            system = system_prod
        else:
            _update_job(job_id, progress=65,
                        message="[Fase 7] Producción directa (sin solvente)...")

        # -------- Bucle de producción ------------------------------------ #
        spf = max(1, steps // nf)
        frames_pdb = []
        frames_positions_nm = []
        analytics = {
            "frames": [], "potential": [], "kinetic": [],
            "total": [], "temp": [], "rmsd": [], "rg": [],
        }
        ref_positions = None

        # Sub-topología del soluto (para escribir PDB multi-modelo)
        solute_topology, _ = _extract_sub_topology(topology, set(solute_indices))

        for i in range(nf):
            check_job_cancelled(job_id)
            simulation.step(spf)

            state = simulation.context.getState(
                getPositions=True, getEnergy=True, enforcePeriodicBox=True,
            )
            pos_nm = state.getPositions(asNumpy=True)
            box_nm = state.getPeriodicBoxVectors(asNumpy=True)
            pe = state.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
            ke = state.getKineticEnergy().value_in_unit(unit.kilocalorie_per_mole)
            te = pe + ke

            # Temperatura a partir de la energía cinética (grados de libertad)
            dof = max(1, 3 * system.getNumParticles() - system.getNumConstraints() - 3)
            temp = (2 * ke) / (
                dof * unit.MOLAR_GAS_CONSTANT_R.value_in_unit(
                    unit.kilocalorie_per_mole / unit.kelvin
                )
            )

            # Analíticas: RMSD (sobre CA) y Rg
            try:
                pos_ang = pos_nm.value_in_unit(unit.angstrom)
                ca_pos = pos_ang[ca_indices]
                if i == 0:
                    ref_positions = ca_pos
                rmsd = float(calc_rmsd(ca_pos, ref_positions)) if ref_positions is not None else 0.0
                rg = float(calc_rg(ca_pos))
            except Exception as e:
                logger.warning(f"[MD {job_id}] analítica frame {i+1} falló: {e}")
                rmsd, rg = 0.0, 0.0

            analytics["frames"].append(i + 1)
            analytics["potential"].append(float(pe) if math.isfinite(pe) else 0.0)
            analytics["kinetic"].append(float(ke) if math.isfinite(ke) else 0.0)
            analytics["total"].append(float(te) if math.isfinite(te) else 0.0)
            analytics["temp"].append(float(temp) if math.isfinite(temp) else 0.0)
            analytics["rmsd"].append(rmsd)
            analytics["rg"].append(rg)

            # Guardar frame como PDB (solo soluto, con PBC aplicado)
            try:
                pos_ang_all = pos_nm.value_in_unit(unit.angstrom)
                box_ang = box_nm.value_in_unit(unit.angstrom)
                solute_pos = pos_ang_all[solute_indices]
                solute_center = solute_pos.mean(axis=0)
                box_center = (box_ang[0] + box_ang[1] + box_ang[2]) / 2.0
                shift = box_center - solute_center
                pos_wrapped = pos_ang_all + shift
                solute_positions_wrapped = pos_wrapped[solute_indices]

                out = StringIO()
                omm_app.PDBFile.writeFile(
                    solute_topology,
                    solute_positions_wrapped * unit.angstrom,
                    out,
                    keepIds=False,
                )
                frames_pdb.append(out.getvalue().replace("END\n", ""))
            except Exception as e:
                # Fallback: guardar sin re-centrado
                logger.error(f"[MD {job_id}] PBC/wrap frame {i+1} falló: {e}")
                out = StringIO()
                omm_app.PDBFile.writeFile(
                    solute_topology,
                    pos_nm[solute_indices],
                    out,
                    keepIds=False,
                )
                frames_pdb.append(out.getvalue().replace("END\n", ""))

            frames_positions_nm.append(pos_nm)
            _update_job(job_id,
                        progress=70 + int((i / nf) * 20),
                        message=f"[Fase 7] Producción frame {i+1}/{nf}")

        # PDB multi-modelo (todos los frames concatenados)
        multi_model_pdb = "".join(
            f"MODEL     {i+1}\n{f}ENDMDL\n" for i, f in enumerate(frames_pdb)
        )

        # -------- MM/GBSA ------------------------------------------------ #
        _update_job(job_id, progress=92,
                    message="[MM/GBSA] Calculando afinidad de unión (OBC2)...")
        mmgbsa_energy = None
        mmgbsa_note = None
        try:
            if has_ligand and protein_indices and ligand_indices and frames_positions_nm:
                # SystemGenerator para GB/OBC2 (no periódico)
                sysgen_gb = SystemGenerator(
                    forcefields=['amber14-all.xml', 'implicit/obc2.xml'],
                    small_molecule_forcefield='openff-2.1.0',
                    forcefield_kwargs={
                        'constraints': None,
                        'rigidWater': False,
                        'removeCMMotion': False,
                    },
                    nonperiodic_forcefield_kwargs={
                        'nonbondedMethod': omm_app.CutoffNonPeriodic,
                        'nonbondedCutoff': 2.0 * unit.nanometer,
                    },
                    molecules=ligand_mols
                )
                mmgbsa_energy, mmgbsa_note = _compute_mmgbsa(
                    topology, frames_positions_nm,
                    protein_indices, ligand_indices,
                    sysgen_gb, platform, ligand_mols, n_samples=5,
                )
            else:
                mmgbsa_note = "MM/GBSA omitido (sin ligando o sin frames)"
        except Exception as e:
            logger.warning(f"[MM/GBSA] cálculo falló: {e}")
            mmgbsa_note = f"Error: {e}"

        # -------- Pipeline de renderizado (GL) --------------------------- #
        lig_resname = valid_res_names[0] if valid_res_names else None
        render_pipeline = {
            "filter_out_resnames": [
                "HOH", "WAT", "SOL", "TIP3", "NA", "CL", "K", "MG", "ZN",
                "CA", "MN", "FE", "CU", "CO", "NI", "CD", "HG",
            ],
            "styles": [
                {
                    "label": "protein_cartoon",
                    "selection": {"backbone": True},
                    "style": {"cartoon": {"color": "spectrum", "opacity": 0.9}},
                },
            ],
            "camera": {
                "zoom_to": {"resn": lig_resname} if lig_resname else {"backbone": True},
                "animation": {"loop": "forward", "reps": 0, "interval": 150},
            },
            "frame_count": nf,
            "has_ligand": has_ligand,
            "ligand_resname": lig_resname,
        }
        if lig_resname:
            # Estilo extra para resaltar el ligando
            render_pipeline["styles"].append({
                "label": "ligand_highlight",
                "selection": {"resn": [lig_resname]},
                "style": {
                    "stick": {"colorscheme": "greenCarbon", "radius": 0.25},
                    "sphere": {"scale": 0.35, "colorscheme": "greenCarbon"},
                },
            })

        pe_last = analytics["potential"][-1] if analytics["potential"] else 0.0

        results = {
            "pdb_result": multi_model_pdb,
            "final_energy": f"{pe_last:.2f} kcal/mol",
            "nframes": nf, "steps": steps,
            "has_ligand": has_ligand,
            "source_docking": source_docking,
            "pose_index": pose_index,
            "pose_score": pose_score,
            "ligand_atoms": ligand_atoms_meta,
            "system_particles": system.getNumParticles(),
            "analytics": analytics,
            "mmgbsa_binding_energy": (
                f"{mmgbsa_energy:.2f} kcal/mol" if mmgbsa_energy is not None else "N/A"
            ),
            "mmgbsa_raw": mmgbsa_energy,
            "mmgbsa_note": mmgbsa_note,
            "gl_render_pipeline": render_pipeline,
        }
        _update_job(job_id, status="done", progress=100,
                    message="MD completada (7 fases + MM/GBSA)", results=results)

    except JobCancelled:
        _update_job(job_id, status="cancelled", message="MD cancelada")
    except Exception as e:
        logger.exception("MD falló")
        _update_job(job_id, status="error", progress=100, message=f"Error MD: {e}")


# ==============================================================================
# DOCKING
# ==============================================================================
def extract_ligand_from_pdb(pdb_text):
    """
    Extrae el ligando de un bloque PDB y lo reetiqueta como ligando
    (registro ``HETATM``, nombre de residuo ``LIG``, cadena ``L``).

    Se usa tanto tras el docking como al preparar el MD a partir de una
    pose. Un ligando PDB "puro" no tiene ``TER``/``END``; aquí se añade
    ``END`` al final.
    """
    lig = []
    for i, line in enumerate((pdb_text or "").splitlines(), 1):
        if not line.startswith(("HETATM", "ATOM  ")):
            continue
        serial = f"{i:5d}"
        line = "HETATM" + serial + line[11:]
        if len(line) >= 20:
            line = line[:17] + "LIG" + line[20:]
        if len(line) > 21:
            line = line[:21] + "L" + line[22:]
        lig.append(line)

    if not lig:
        return ""
    lig.append("END")
    return "\n".join(lig)


def create_labeled_complex(protein_pdb, ligand_pose, ligand_smiles, wd):
    """
    Crea un PDB "complejo etiquetado" (proteína + pose del ligando) con
    nombres de cadena/residuo distintos para que PyMOL/PLIP los distingan:

        - Proteína : cadena ``P`` (o ``H`` si es un grupo hemo).
        - Ligando  : cadena ``L`` y nombre de residuo derivado de la
                     fórmula molecular del SMILES (si es corta), o ``LIG``.

    Devuelve:
        (contenido_pdb, res_name)
    """
    res_name = "LIG"
    try:
        mol_rdkit = Chem.MolFromSmiles(ligand_smiles)
        if mol_rdkit:
            formula = Chem.rdMolDescriptors.calcMolFormula(mol_rdkit)
            # Si la fórmula es corta (≤3 caracteres) la usamos como nombre
            if 0 < len(formula) <= 3:
                res_name = formula
    except Exception:
        pass

    # Reetiquetar la proteína
    p_lines = []
    for line in protein_pdb.splitlines():
        if line.startswith(('ATOM', 'HETATM')):
            if "HEM" in line:
                if len(line) > 21:
                    line = line[:21] + 'H' + line[22:]
            else:
                if len(line) > 21 and line[21] == ' ':
                    line = line[:21] + 'P' + line[22:]
            p_lines.append(line)

    # Reetiquetar el ligando
    l_lines = []
    for line in ligand_pose.splitlines():
        if line.startswith(('HETATM', 'ATOM')):
            if len(line) >= 20:
                line = line[:17] + f"{res_name:<3}"[:3] + line[20:]
            if len(line) > 21:
                line = line[:21] + 'L' + line[22:]
            l_lines.append(line)

    content = '\n'.join(p_lines + ['TER'] + l_lines + ['END'])
    try:
        (wd / "complex_labeled.pdb").write_text(content)
    except Exception:
        pass
    return content, res_name


def run_docking_job(job_id, pdb_text, ligand_smiles, exhaustiveness=8, cpu_cores=2):
    """
    Ejecuta un docking molecular con AutoDock Vina.

    Pipeline:
        1. Convertir proteína PDB → PDBQT (obabel, protonación a pH 7.4).
        2. Calcular el centroide de la proteína (caja de búsqueda 40³ Å).
        3. Convertir SMILES → PDBQT (obabel --gen3d).
        4. Ejecutar ``vina`` con exhaustiveness y CPUs indicados.
        5. Parsear los scores de las poses (``REMARK VINA RESULT:``).
        6. Convertir el PDBQT de salida a PDB y separar poses.
        7. Calcular Ki por pose (``calcular_ki``) y generar un PDB
           "complejo etiquetado".

    Resultado en ``JOBS[job_id]["results"]``:
        scores, ki_values, poses, protein_pdb, ligand_only, complex_pdb,
        chemical_name, pose_atom_counts, analytics.
    """
    try:
        check_job_cancelled(job_id)
        wd = JOBS_DIR / job_id
        wd.mkdir(exist_ok=True)
        _update_job(job_id, status="running", progress=10, message="Analizando...")

        # Escribir proteína
        protein_pdb = wd / "protein.pdb"
        protein_pdb.write_text(pdb_text)
        protein_pdbqt = wd / "protein.pdbqt"

        # Centroide de la proteína (centro de la caja de búsqueda)
        c = [0.0, 0.0, 0.0]
        if BIO_AVAILABLE:
            try:
                st = PDBParser(QUIET=True).get_structure("p", str(protein_pdb))
                coords = [a.get_coord() for m in st for ch in m for r in ch for a in r]
                if coords:
                    c = np.mean(coords, axis=0).tolist()
            except Exception:
                pass

        # Proteína → PDBQT (mantener residuos, protonar a pH 7.4)
        subprocess.run(
            ["obabel", str(protein_pdb), "-O", str(protein_pdbqt), "-xr", "-p", "7.4"],
            check=True, capture_output=True,
        )

        # SMILES → PDBQT (con generación 3D + H)
        smi = wd / "ligand.smi"
        smi.write_text(ligand_smiles)
        lig_pdbqt = wd / "ligand.pdbqt"
        subprocess.run(
            ["obabel", str(smi), "-O", str(lig_pdbqt), "--gen3d", "-h", "-p", "7.4"],
            check=True, capture_output=True,
        )

        # Normalizar exhaustiveness
        try:
            exh = int(exhaustiveness)
        except Exception:
            exh = 8

        # Ejecutar Vina
        out_pdbqt = wd / "docking_out.pdbqt"
        cmd = [
            "vina",
            "--receptor", str(protein_pdbqt),
            "--ligand", str(lig_pdbqt),
            "--center_x", str(c[0]), "--center_y", str(c[1]), "--center_z", str(c[2]),
            "--size_x", "40", "--size_y", "40", "--size_z", "40",
            "--cpu", str(cpu_cores), "--exhaustiveness", str(exh),
            "--out", str(out_pdbqt),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"Vina: {r.stderr}")

        # Parsear scores de las poses
        scores = []
        with open(out_pdbqt) as f:
            for line in f:
                if line.startswith("REMARK VINA RESULT:"):
                    parts = line.split()
                    try:
                        scores.append(float(parts[3]))
                    except Exception:
                        pass
        if not scores:
            raise RuntimeError("Vina no generó poses.")

        # PDBQT → PDB y separar poses
        out_pdb = wd / "docking_out.pdb"
        subprocess.run(["obabel", str(out_pdbqt), "-O", str(out_pdb)],
                       check=True, capture_output=True)

        poses = _split_models_from_file(out_pdb)
        if not poses:
            poses = _split_models_from_file(out_pdbqt)
        if not poses:
            raise RuntimeError("No se pudieron parsear las poses.")

        atom_counts = [
            sum(1 for l in p.splitlines() if l.startswith(("ATOM", "HETATM")))
            for p in poses
        ]
        ki_values = [calcular_ki(s) for s in scores]

        ligand_only = extract_ligand_from_pdb(poses[0])
        complex_pdb, chem_name = create_labeled_complex(pdb_text, poses[0], ligand_smiles, wd)

        _update_job(
            job_id, status="done", progress=100,
            message=f"Docking completado ({len(poses)} poses)",
            results={
                "score": f"{scores[0]} kcal/mol",
                "scores": scores,
                "ki_values": ki_values,
                "poses": poses,
                "protein_pdb": pdb_text,
                "ligand_only": ligand_only,
                "complex_pdb": complex_pdb,
                "chemical_name": chem_name,
                "pose_atom_counts": atom_counts,
                "analytics": {
                    "frames": list(range(1, len(scores) + 1)),
                    "scores": scores,
                    "ki": ki_values,
                    "rmsd": [0.0] * len(scores),
                },
            },
        )

    except JobCancelled:
        _update_job(job_id, status="cancelled", message="Cancelado")
    except Exception as e:
        logger.exception("Docking falló")
        _update_job(job_id, status="error", progress=100, message=f"Error Docking: {e}")


# ==============================================================================
# ENDPOINTS REST
# ==============================================================================
@bp.route("/api/fetch_pdb/<pdb_id>")
def fetch_pdb(pdb_id):
    """
    Descarga un PDB de RCSB por su identificador.

    Salida:
        - Éxito: { "pdb_text": "<contenido>" }
        - Error: { "error": "..." } (404/500)
    """
    try:
        r = requests.get(f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb", timeout=15)
        if r.status_code == 200:
            return jsonify({"pdb_text": r.text})
        return jsonify({"error": f"No encontrado {pdb_id}"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/api/docking", methods=["POST"])
def start_docking():
    """
    Lanza un docking molecular en segundo plano.

    Entrada JSON:
        {
            "pdb_text":      "<PDB proteína>",
            "smiles":        "<SMILES ligando>",
            "exhaustiveness": 8,
            "cpu_cores":     2
        }

    Salida JSON:
        { "job_id": "<uuid>" }
    """
    data = request.json or {}
    job_id = str(uuid.uuid4())
    exhaustiveness = data.get("exhaustiveness", 8)
    cpu_cores = int(data.get("cpu_cores", 2))

    _update_job(job_id, status="running", progress=0, task="docking",
                exhaustiveness=exhaustiveness, cpu_cores=cpu_cores)

    threading.Thread(
        target=run_docking_job,
        args=(job_id, data.get("pdb_text"), data.get("smiles"),
              exhaustiveness, cpu_cores),
    ).start()
    return jsonify({"job_id": job_id})


@bp.route("/api/md", methods=["POST"])
def start_md():
    """
    Lanza una MD proteína-ligando en segundo plano.

    Entrada JSON:
        {
            "pdb_text":    "<PDB proteína>",
            "ligand_pdb":  "<PDB ligando>" (opcional),
            "nframes":     10,
            "steps":       5000,
            "solvent":     "tip3p",
            "forcefield":  "amber14-all.xml",
            "cpu_cores":   2,
            "box_shape":   "octahedron"
        }

    Salida JSON:
        { "job_id": "<uuid>" }
    """
    data = request.json or {}
    job_id = str(uuid.uuid4())
    nframes = int(data.get("nframes", 10))
    steps = int(data.get("steps", 5000))
    solvent = data.get("solvent", "tip3p")
    forcefield = data.get("forcefield", "amber14-all.xml")
    cpu_cores = int(data.get("cpu_cores", 2))
    box_shape = data.get("box_shape", "octahedron")

    _update_job(job_id, status="running", progress=0, task="md",
                nframes=nframes, steps=steps, solvent=solvent,
                forcefield=forcefield, cpu_cores=cpu_cores, box_shape=box_shape)

    threading.Thread(
        target=run_md_job,
        args=(job_id, data.get("pdb_text"), data.get("ligand_pdb"),
              steps, nframes, solvent, forcefield, cpu_cores, box_shape),
    ).start()
    return jsonify({"job_id": job_id})


def _preview_pose(pose_text, max_lines=3, max_chars=200):
    """
    Devuelve una vista previa corta de una pose (para mensajes de error
    y debug): las primeras ``max_lines`` líneas, unidas por ' | '.
    """
    lines = (pose_text or "").splitlines()[:max_lines]
    preview = " | ".join(l.strip() for l in lines if l.strip())
    return preview[:max_chars] + ("..." if len(preview) > max_chars else "")


def _recover_pose_from_disk(docking_job_id, pose_index):
    """
    Intenta recuperar una pose desde el disco cuando en ``JOBS`` la pose
    viene vacía (por ejemplo, tras reiniciar el servidor).

    Busca en ``docking_out.pdb`` y ``docking_out.pdbqt`` dentro de la
    carpeta del job de docking. Devuelve el texto de la pose o "".
    """
    wd = JOBS_DIR / docking_job_id
    for fname in ("docking_out.pdb", "docking_out.pdbqt"):
        path = wd / fname
        if not path.exists():
            continue
        poses = _split_models_from_file(path)
        if pose_index < len(poses):
            cand = poses[pose_index]
            n = sum(1 for l in cand.splitlines() if l.startswith(("ATOM", "HETATM")))
            if n > 0:
                return cand
    return ""


@bp.route("/api/md_from_docking", methods=["POST"])
def md_from_docking():
    """
    Inicia una MD a partir de una pose de un docking previamente completado.

    Entrada JSON:
        {
            "docking_job_id": "<uuid>",
            "pose_index":     0,
            "nframes":        10,
            "steps":          5000,
            "solvent":        "tip3p",
            "forcefield":     "amber14-all.xml",
            "cpu_cores":      2,
            "box_shape":      "octahedron"
        }

    Salida:
        - Éxito: JSON con job_id del MD + metadatos de la pose.
        - Error: JSON con mensaje y (si aplica) debug.
    """
    data = request.json or {}
    dj = data.get("docking_job_id")

    # Validar existencia y estado del docking
    with JOBS_LOCK:
        d = JOBS.get(dj) if dj else None
    if not d or d.get("status") != "done":
        return jsonify({"error": "Docking no completado o inexistente"}), 400

    r = d.get("results", {}) or {}
    poses = r.get("poses", []) or []
    if not poses:
        return jsonify({"error": "El docking no contiene poses"}), 400

    # Validar índice de pose
    try:
        pose_index = int(data.get("pose_index", 0))
    except (TypeError, ValueError):
        pose_index = 0
    if not (0 <= pose_index < len(poses)):
        return jsonify({"error": f"pose_index={pose_index} fuera de rango"}), 400

    protein_pdb = r.get("protein_pdb")
    if not protein_pdb:
        return jsonify({"error": "No hay estructura de proteína"}), 400

    # Recuperar la pose (y fallback a disco si está vacía)
    raw_pose = poses[pose_index] or ""
    raw_atom_count = sum(1 for l in raw_pose.splitlines() if l.startswith(("ATOM  ", "HETATM")))
    recovered_from = None
    if raw_atom_count == 0:
        raw_pose = _recover_pose_from_disk(dj, pose_index)
        raw_atom_count = sum(1 for l in raw_pose.splitlines() if l.startswith(("ATOM  ", "HETATM")))
        if raw_atom_count > 0:
            recovered_from = "disk"

    if raw_atom_count == 0:
        return jsonify({
            "error": f"La pose {pose_index+1} está vacía.",
            "debug": {"preview": _preview_pose(raw_pose)},
        }), 400

    # Normalizar la pose a un PDB de ligando
    ligand_pdb = extract_ligand_from_pdb(raw_pose)
    n_lig = sum(1 for l in ligand_pdb.splitlines() if l.startswith("HETATM"))
    if n_lig == 0:
        return jsonify({"error": f"La pose {pose_index+1} no tiene ligando tras normalizar."}), 400

    n_prot = sum(1 for l in protein_pdb.splitlines() if l.startswith(("ATOM  ", "HETATM")))
    scores = r.get("scores", []) or []
    pose_score = scores[pose_index] if pose_index < len(scores) else None

    # Parámetros del MD
    nframes = int(data.get("nframes", 10))
    steps = int(data.get("steps", 5000))
    solvent = data.get("solvent", "tip3p")
    forcefield = data.get("forcefield", "amber14-all.xml")
    cpu_cores = int(data.get("cpu_cores", 2))
    box_shape = data.get("box_shape", "octahedron")

    job_id = str(uuid.uuid4())
    _update_job(
        job_id, status="running", progress=0, task="md_from_docking",
        source_docking=dj, pose_index=pose_index, pose_score=pose_score,
        ligand_atoms=n_lig, protein_atoms=n_prot,
        n_poses=len(poses), recovered_from=recovered_from,
        nframes=nframes, steps=steps, solvent=solvent,
        forcefield=forcefield, cpu_cores=cpu_cores, box_shape=box_shape,
    )

    threading.Thread(
        target=run_md_job,
        args=(job_id, protein_pdb, ligand_pdb, steps, nframes,
              solvent, forcefield, cpu_cores, box_shape),
    ).start()

    return jsonify({
        "job_id": job_id,
        "pose_index": pose_index,
        "n_poses": len(poses),
        "ligand_atoms": n_lig,
        "protein_atoms": n_prot,
        "pose_score": pose_score,
        "complex_confirmed": True,
        "recovered_from_disk": recovered_from == "disk",
        "message": f"MD iniciada • pose {pose_index+1}/{len(poses)} • "
                   f"{n_lig} át ligando • {n_prot} át proteína",
    })


@bp.route("/api/alphafold", methods=["POST"])
def start_alphafold():
    """
    Lanza la predicción de estructura (AlphaFold DB / ESMFold) en segundo
    plano.

    Entrada JSON:
        { "sequence": "<secuencia de aminoácidos>" }

    Salida JSON:
        { "job_id": "<uuid>" }
    """
    data = request.json or {}
    seq = data.get("sequence", "").strip()
    if not seq:
        return jsonify({"error": "Secuencia vacía"}), 400

    job_id = str(uuid.uuid4())
    _update_job(job_id, status="running", progress=0, task="alphafold")
    threading.Thread(target=run_alphafold_job, args=(job_id, seq)).start()
    return jsonify({"job_id": job_id})


@bp.route("/api/export_to_pymol", methods=["POST"])
def export_to_pymol():
    """
    Exporta el resultado de un job a un fichero PDB "para PyMOL".

    Tipos de exportación:
        - docking    : ``complex_pdb`` del resultado.
        - md/alphafold: ``pdb_result`` del resultado.
        - dft        : geometría XYZ → PDB (todos los átomos como HETATM).

    Entrada JSON:
        { "job_id": "<uuid>", "type": "auto"|"docking"|"md"|"alphafold"|"dft" }

    Salida JSON:
        { "success": true, "file_path": "...", "file_url": "/jobs/<id>/..." }
        o { "error": "..." } (404/500).
    """
    data = request.json or {}
    jid = data.get("job_id")
    t = data.get("type", "auto")

    with JOBS_LOCK:
        job = JOBS.get(jid) if jid else None
    if not job:
        return jsonify({"error": "No encontrado"}), 404

    wd = JOBS_DIR / jid
    exp = None

    # Auto-detección del tipo a partir del "task"
    if t == "auto":
        task_str = str(job.get("task", ""))
        if "docking" in task_str:
            t = "docking"
        elif "md" in task_str:
            t = "md"
        elif "alphafold" in task_str:
            t = "alphafold"
        else:
            t = "dft"

    if t == "docking":
        r = job.get("results", {})
        if r.get("complex_pdb"):
            exp = wd / "for_pymol_complex.pdb"
            exp.write_text(r["complex_pdb"])
    elif t in ("md", "alphafold"):
        r = job.get("results", {})
        if r.get("pdb_result"):
            exp = wd / f"for_pymol_{t}.pdb"
            exp.write_text(r["pdb_result"])
    elif t == "dft" and job.get("geometry"):
        xyz_lines = job["geometry"].split("\n")
        pdb = "HEADER    DFT\n"
        for i, line in enumerate(xyz_lines[2:], 1):
            p = line.split()
            if len(p) >= 4:
                pdb += (
                    f"HETATM{i:5d} {p[0]:>2s}   LIG A   1    "
                    f"{float(p[1]):8.3f}{float(p[2]):8.3f}{float(p[3]):8.3f}"
                    f"  1.00  0.00          {p[0]:>2s}\n"
                )
        exp = wd / "for_pymol_dft.pdb"
        exp.write_text(pdb + "END\n")

    if exp and exp.exists():
        return jsonify({
            "success": True,
            "file_path": str(exp),
            "file_url": f"/jobs/{jid}/{exp.name}",
        })
    return jsonify({"error": "No se pudo generar archivo"}), 500


@bp.route("/api/pymol_command", methods=["POST"])
def pymol_command():
    """
    Envía comandos arbitrarios a PyMOL vía XML-RPC.

    Entrada JSON (todo opcional):
        {
            "file_path": "<ruta local a cargar>",
            "pdb_text":  "<PDB inline>",
            "commands":  "<comandos separados por saltos de línea>"
        }

    Prioridad de carga:
        1. ``file_path`` si está presente.
        2. ``pdb_text`` en un fichero temporal si no lo está.

    Salida:
        { "success": true, "log": "..." }  o  { "error": "..." }
    """
    data = request.json or {}
    commands = data.get("commands", "")
    file_path = data.get("file_path", "")
    pdb_text = data.get("pdb_text", "")

    try:
        server = xmlrpc.client.ServerProxy('http://127.0.0.1:9123', allow_none=True)
        log = []

        if file_path:
            server.do(f"load {file_path}")
            log.append(f"Cargado: {file_path}")
        elif pdb_text:
            tid = str(uuid.uuid4())
            td = JOBS_DIR / tid
            td.mkdir(exist_ok=True)
            tf = td / "input.pdb"
            tf.write_text(pdb_text)
            server.do(f"load {tf}")
            log.append("Cargado desde texto")

        if commands:
            for line in commands.strip().split('\n'):
                if line.strip() and not line.startswith('#'):
                    server.do(line.strip())
                    log.append(f"> {line.strip()}")

        return jsonify({"success": True, "log": "Enviado.\n" + "\n".join(log)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
