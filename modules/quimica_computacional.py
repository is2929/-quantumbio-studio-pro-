"""
Módulo de Química Computacional de "QuantumBio Studio Pro".

Este módulo registra un Blueprint de Flask (``quimica_computacional``) que
expone la API y los motores de cálculo DFT basados en:

    - **PySCF**  : motor nativo Python (SCF, propiedades, orbitales, Hessiano).
    - **ORCA**   : binario externo invocado como subproceso (Opt/Freq).
    - **BSE**    : opcional, "Basis Set Exchange" para obtener bases de forma
                   dinámica (si no está disponible, se usan bases nativas).

Flujo típico desde el cliente web
---------------------------------
    1. ``/api/preview``             — SMILES → XYZ (para previsualización 3D).
    2. ``/api/upload_structure``    — Subir .xyz / .inp / .pdb.
    3. ``/api/list_basis_sets``     — Listar bases disponibles.
    4. ``/api/calcular``            — Lanzar cálculo DFT (thread en segundo plano).
    5. ``/api/estado/<job_id>``     — (definido en main.py) consultar progreso.

Notas de implementación
-----------------------
    - Todos los nombres de ``core`` (utilidades y estado global) se importan
      aquí con ``globals().update(...)`` para no tener que prefijar cada uno
      con ``_core.``.
    - Los cálculos se ejecutan en hilos (``threading.Thread``) y actualizan el
      diccionario global ``JOBS`` a través de ``_update_job``.
"""

# Importa el módulo "core" bajo el alias _core y luego vuelca todos sus
# nombres públicos (los que no empiezan por "__") en el espacio global de
# este módulo. Así podemos usar JOBS, JOBS_DIR, JOBS_LOCK, ACTIVE_PROCESSES,
# logger, bse, gto, dft, scf, cubegen, thermo, np, re, subprocess, time,
# uuid, threading, JobCancelled, check_job_cancelled, _update_job,
# _build_vibrations, Chem, AllChem, rdDetermineBonds, ATOMIC_Z,
# FUNCTIONALS, NATIVE_ORCA_BASES, NATIVE_PYSCF_BASES, BSE_AVAILABLE,
# BSE_CACHE_FILE, PYSCF_AVAILABLE, ORCA_BINARY, ORCA_LIB_DIR, etc.,
# directamente sin prefijo.
import core as _core
globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})

from flask import Blueprint

# Blueprint que agrupa todas las rutas de este módulo.
bp = Blueprint("quimica_computacional", __name__)


# --------------------------------------------------------------------------- #
# Validación de métodos (funcional / base)
# --------------------------------------------------------------------------- #
def validate_methods(functional, basis):
    """
    Comprueba que la combinación funcional + base sea válida.

    - El funcional debe estar en el conjunto ``FUNCTIONALS`` (definido en core).
    - Si ``BSE_AVAILABLE`` es False (no hay acceso a Basis Set Exchange), la
      base debe pertenecer a los conjuntos nativos de ORCA o PySCF.

    Retorna:
        (True, "") si es válido, o (False, mensaje_de_error) si no lo es.
    """
    if functional.upper() not in FUNCTIONALS:
        return False, f"Funcional '{functional}' no soportado."

    if not BSE_AVAILABLE:
        if basis.upper() not in NATIVE_ORCA_BASES and basis.upper() not in NATIVE_PYSCF_BASES:
            return False, f"Base '{basis}' no nativa (BSE no instalado)."

    return True, ""


# --------------------------------------------------------------------------- #
# Endpoint: listado de bases disponibles
# --------------------------------------------------------------------------- #
@bp.route("/api/list_basis_sets", methods=["GET"])
def list_basis_sets():
    """
    Devuelve la lista de bases disponibles.

    Estrategia (de mayor a menor prioridad):
        1. Si existe un fichero de caché (``BSE_CACHE_FILE``), devolverlo.
        2. Si BSE no está disponible, devolver solo las bases nativas
           (ORCA + PySCF) combinadas.
        3. Si BSE sí está disponible, consultar sus metadatos, cachear el
           resultado y devolverlo.
        4. Si la consulta a BSE falla, devolver el fallback nativo.
    """
    # 1. Caché
    if BSE_CACHE_FILE.exists():
        try:
            with open(BSE_CACHE_FILE, 'r') as f:
                return jsonify(json.load(f))
        except Exception:
            # Si la caché está corrupta, se ignora y se continúa.
            pass

    # Fallback común: bases nativas
    fallback = sorted(list(NATIVE_ORCA_BASES) + list(NATIVE_PYSCF_BASES))

    # 2. Sin BSE → solo nativas
    if not BSE_AVAILABLE:
        return jsonify({
            "basis_sets": fallback,
            "source": "native",
            "count": len(fallback),
        })

    # 3. Consultar BSE y cachear
    try:
        meta = bse.get_metadata()
        all_bases = sorted(meta.get('basis_set_names', []))
        resp = {"basis_sets": all_bases, "source": "bse", "count": len(all_bases)}
        with open(BSE_CACHE_FILE, 'w') as f:
            json.dump(resp, f)
        return jsonify(resp)
    except Exception:
        # 4. Si BSE falla, devolver nativas
        return jsonify({
            "basis_sets": fallback,
            "source": "fallback",
            "count": len(fallback),
        })


# --------------------------------------------------------------------------- #
# Utilidades: SMILES → XYZ y análisis de geometría XYZ
# --------------------------------------------------------------------------- #
def smiles_to_xyz(smiles, seed=42):
    """
    Convierte un SMILES en un bloque XYZ 3D.

    Pasos:
        1. Parsear SMILES con RDKit.
        2. Añadir hidrógenos explícitos.
        3. Generar embedding 3D con ETKDGv3 (semilla fija → reproducible).
        4. Optimizar la geometría con MMFF94.
        5. Serializar las coordenadas a formato XYZ.

    Parámetros:
        smiles : str
        seed   : int, semilla del generador conformacional.

    Retorna:
        str con el bloque XYZ (primera línea: nº átomos; segunda: comentario).

    Lanza:
        ValueError si el SMILES es inválido.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"SMILES inválido")

    mol = Chem.AddHs(mol)

    # Embedding 3D reproducible
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    AllChem.EmbedMolecule(mol, params)
    AllChem.MMFFOptimizeMolecule(mol)

    # Serializar a XYZ
    conf = mol.GetConformer()
    lines = [str(mol.GetNumAtoms()), f"Generated: {smiles}"]
    for atom in mol.GetAtoms():
        p = conf.GetAtomPosition(atom.GetIdx())
        lines.append(f"{atom.GetSymbol():2s} {p.x:12.6f} {p.y:12.6f} {p.z:12.6f}")
    return "\n".join(lines)


def analyze_xyz(xyz_text, charge=0):
    """
    Analiza un bloque XYZ y extrae información básica de la molécula:

        - Fórmula molecular (con multiplicidades tipo Hill).
        - Número de átomos.
        - Número total de electrones (usando el número atómico y la carga).
        - SMILES reconstruido (si RDKit puede determinarlo).

    Parámetros:
        xyz_text : str, contenido en formato XYZ.
        charge   : int, carga total de la molécula.

    Retorna:
        dict con claves: ``formula``, ``n_atoms``, ``n_electrons``, ``smiles``.
    """
    lines = [l for l in xyz_text.strip().splitlines() if l.strip()]
    if not lines:
        return {}

    n_atoms = int(lines[0].split()[0])

    # Contar átomos por elemento
    counts = {}
    for line in lines[2:2 + n_atoms]:
        el = line.split()[0].capitalize()
        counts[el] = counts.get(el, 0) + 1

    # Electrones totales: suma Z*n - carga
    n_el = sum(ATOMIC_Z.get(el, 0) * n for el, n in counts.items()) - charge

    # Intento de reconstruir el SMILES con RDKit
    smiles = None
    try:
        mol = Chem.MolFromXYZBlock("\n".join(lines[:n_atoms + 2]))
        if mol:
            rdDetermineBonds.DetermineBonds(mol, charge=charge)
            smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
    except Exception:
        # Si falla la percepción de enlaces, dejamos smiles=None.
        pass

    # Fórmula estilo Hill simplificada (sin subíndices para n=1)
    formula = "".join(
        el if counts[el] == 1 else f"{el}{counts[el]}"
        for el in sorted(counts)
    )

    return {
        "formula": formula,
        "n_atoms": n_atoms,
        "n_electrons": n_el,
        "smiles": smiles,
    }


# ==============================================================================
# NUEVO: PARSER .INP (ORCA / Gaussian) → XYZ
# ==============================================================================
def parse_inp_to_xyz(inp_text):
    """
    Convierte un archivo .inp (ORCA o Gaussian) a formato XYZ.

    Soporta:
      - ORCA: bloques  * xyz Q M  /  * int Q M  /  * xyzfile Q M file.xyz
      - Gaussian: línea "charge multiplicity" seguida de coordenadas cartesianas

    Estrategia:
        1. Buscar primero un bloque ORCA (``* xyz ...``, ``* int ...``,
           ``* xyzfile ...``). Se recorren los átomos hasta encontrar
           ``*`` o ``%`` o línea vacía.
        2. Si no hay bloque ORCA, buscar el patrón Gaussian: una línea con
           dos enteros (carga y multiplicidad) seguida de coordenadas.
        3. Si nada funciona, lanzar ValueError con un mensaje explicativo
           (incluye el caso especial de ``* xyzfile`` con archivo externo,
           que no podemos resolver sin el .xyz).

    Retorna:
        (xyz_text, charge, multiplicity)
    """
    lines = inp_text.splitlines()
    atoms = []
    charge, multiplicity = 0, 1
    external_xyz = None

    # -------------------------- Intento 1: ORCA ---------------------------- #
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith('*'):
            header = stripped[1:].strip()
            parts = header.split()
            is_xyzfile = False

            # Caso "* xyzfile Q M file.xyz"
            if len(parts) >= 1 and parts[0].lower() == 'xyzfile':
                is_xyzfile = True
                if len(parts) >= 3:
                    try:
                        charge = int(parts[1])
                        multiplicity = int(parts[2])
                    except (ValueError, IndexError):
                        pass
                if len(parts) >= 4:
                    external_xyz = parts[3]

            # Caso "* xyz Q M" o "* int Q M"
            elif len(parts) >= 3 and parts[0].lower() in ('xyz', 'int'):
                try:
                    charge = int(parts[1])
                    multiplicity = int(parts[2])
                except (ValueError, IndexError):
                    charge, multiplicity = 0, 1

            # Leer coordenadas hasta encontrar '*' / '%' / vacío
            if not is_xyzfile:
                j = i + 1
                while j < len(lines):
                    l = lines[j].strip()
                    if not l or l.startswith('*') or l.startswith('%'):
                        break
                    p = l.split()
                    if len(p) >= 4:
                        try:
                            x, y, z = float(p[1]), float(p[2]), float(p[3])
                            sym = p[0].capitalize()
                            if sym.isalpha():
                                atoms.append((sym, x, y, z))
                        except ValueError:
                            pass
                    j += 1
                if atoms:
                    break
                # Bloque sin coordenadas válidas: continuar buscando
                i = j
                continue
        i += 1

    # ------------------------ Intento 2: Gaussian -------------------------- #
    if not atoms:
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Ignorar comentarios y directivas de Gaussian
            if not stripped or stripped.startswith(('#', '!', '%')):
                continue
            parts = stripped.split()
            if len(parts) == 2:
                try:
                    c = int(parts[0]); m = int(parts[1])
                except ValueError:
                    continue

                tmp_atoms = []
                for j in range(i + 1, len(lines)):
                    l = lines[j].strip()
                    if not l:
                        break
                    p = l.split()
                    if len(p) >= 4:
                        try:
                            x, y, z = float(p[1]), float(p[2]), float(p[3])
                            sym = p[0].capitalize()
                            if sym.isalpha():
                                tmp_atoms.append((sym, x, y, z))
                        except ValueError:
                            break
                if tmp_atoms:
                    atoms = tmp_atoms
                    charge, multiplicity = c, m
                    break

    # ---------------------- Sin coordenadas válidas ------------------------ #
    if not atoms:
        if external_xyz:
            raise ValueError(
                f"El .inp usa '* xyzfile' con archivo externo '{external_xyz}'. "
                f"Sube directamente el archivo .xyz."
            )
        raise ValueError("No se encontraron coordenadas válidas en el archivo .inp")

    # Serializar a XYZ
    xyz_lines = [
        str(len(atoms)),
        f"Converted from INP (Q={charge}, M={multiplicity})",
    ]
    for sym, x, y, z in atoms:
        xyz_lines.append(f"{sym:2s} {x:12.6f} {y:12.6f} {z:12.6f}")
    return "\n".join(xyz_lines), charge, multiplicity


# --------------------------------------------------------------------------- #
# Motor PySCF
# --------------------------------------------------------------------------- #
def ejecutar_pyscf(job_id, xyz_text, charge, multiplicity, basis, functional, task, cpu_cores=2):
    """
    Ejecuta un cálculo DFT con PySCF.

    Flujo:
        1. Preparar entorno (OMP threads).
        2. Construir ``gto.Mole`` con la base pedida (o fallback def2-SVP).
        3. Ejecutar SCF (RKS/UKS o RHF/UHF).
        4. Extraer propiedades (energía, HOMO/LUMO/gap, dipolo).
        5. Generar cubos de orbitales moleculares.
        6. Si task incluye freq → calcular Hessiano y análisis armónico.

    Parámetros:
        job_id        : str, identificador del trabajo (usado por _update_job).
        xyz_text      : str, geometría molecular en formato XYZ.
        charge        : int
        multiplicity  : int (spin = multiplicity - 1)
        basis         : str
        functional    : str ("HF" o funcional DFT)
        task          : str ("single", "freq", "opt_freq", ...)
        cpu_cores     : int

    El estado del job se actualiza paso a paso en ``JOBS`` a través de
    ``_update_job``. En caso de cancelación se marca ``cancelled``; en caso
    de error se marca ``error``.
    """
    try:
        # Limitar hilos BLAS/OpenMP al nº de CPUs asignadas
        os.environ["OMP_NUM_THREADS"] = str(cpu_cores)

        # Permitir cancelación temprana
        check_job_cancelled(job_id)

        if not PYSCF_AVAILABLE:
            raise RuntimeError("PySCF no instalado.")

        _update_job(job_id, status="running", progress=10,
                    message=f"Construyendo molécula ({cpu_cores} núcleos)...")

        wd = JOBS_DIR / job_id
        wd.mkdir(exist_ok=True)
        lines = xyz_text.strip().splitlines()

        # PySCF espera "Sym x y z; Sym x y z; ..."
        atom_block = "; ".join(l.strip() for l in lines[2:])

        # Intentar obtener base en formato PySCF desde BSE (si está disponible)
        basis_arg = basis
        ext_basis = None
        if BSE_AVAILABLE:
            try:
                ext_basis = bse.get_basis(basis, fmt='pyscf')
            except Exception:
                pass
        if ext_basis is not None:
            basis_arg = ext_basis

        # Construir molécula (con fallback a def2-SVP si la base falla)
        try:
            mol = gto.Mole(atom=atom_block, charge=charge, spin=multiplicity - 1,
                           basis=basis_arg, verbose=0)
            mol.build()
        except Exception:
            mol = gto.Mole(atom=atom_block, charge=charge, spin=multiplicity - 1,
                           basis="def2-SVP", verbose=0)
            mol.build()

        # Selección de método: HF o funcional DFT (RKS/UKS según spin)
        mf = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
        if functional.upper() == "HF":
            mf = scf.RHF(mol) if mol.spin == 0 else scf.UHF(mol)
        else:
            mf.xc = functional

        # -------------------------- SCF ----------------------------------- #
        _update_job(job_id, progress=30, message="Ejecutando SCF...")
        mf.kernel()
        check_job_cancelled(job_id)
        _update_job(job_id, progress=50, message="Extrayendo propiedades...")

        # ---------------------- Propiedades ------------------------------- #
        props = {
            "energy": f"{mf.e_tot:.6f}",
            "converged": "Sí" if mf.converged else "No",
        }
        mo_e = np.array(mf.mo_energy).flatten()
        mo_o = np.array(mf.mo_occ).flatten()
        h, l = np.where(mo_o > 0)[0], np.where(mo_o == 0)[0]
        if len(h) > 0 and len(l) > 0:
            # Conversión Hartree → eV (1 Ha = 27.2114 eV)
            he = mo_e[max(h)] * 27.2114
            le = mo_e[min(l)] * 27.2114
            props["homo"] = f"{he:.3f} eV"
            props["lumo"] = f"{le:.3f} eV"
            props["gap"] = f"{le - he:.3f} eV"
        else:
            props["homo"] = props["lumo"] = props["gap"] = "N/A"

        try:
            dip = mf.dip_moment(unit='Debye')
            props["dipole"] = f"{np.linalg.norm(dip):.3f} D"
        except Exception:
            props["dipole"] = "N/A"

        # ---------------------- Orbitales (CUBE) --------------------------- #
        _update_job(job_id, progress=60, message="Generando orbitales...")
        cube_files = {}
        for idx in range(mol.nao_nr()):
            check_job_cancelled(job_id)
            try:
                cubegen.orbital(mol, str(wd / f"mo_{idx}.cube"), mf.mo_coeff[:, idx])
                cube_files[str(idx)] = f"/jobs/{job_id}/mo_{idx}.cube"
            except Exception:
                # Si un orbital concreto falla, se ignora y se continúa.
                pass

        # ---------------------- Frecuencias (opcional) --------------------- #
        vibraciones, freqs_list = [], []
        if task in ("freq", "opt_freq"):
            _update_job(job_id, progress=80, message="Calculando Hessiano...")
            try:
                hesser = scf.hf.Hessian(mf) if functional.upper() == "HF" else mf.Hessian()
                h2 = hesser.kernel()
                res = thermo.harmonic_analysis(mol, h2)
                freqs, modes = res['freq_wavenumber'], res['norm_mode']
                freqs_list = freqs.tolist()

                # Base XYZ sin comentario para construir modos de vibración
                xyz_base = "\n".join([" ".join(l.split()[:4]) for l in lines[2:]])
                n_atoms = len(xyz_base.strip().splitlines())
                vibraciones = _build_vibrations(xyz_base, freqs, modes, n_atoms)
            except Exception as e:
                logger.error(f"Error Hessiano: {e}")

        # ---------------------- Final OK ----------------------------------- #
        check_job_cancelled(job_id)
        _update_job(job_id, status="done", progress=100, message="Completado",
                    geometry=xyz_text,
                    freqs=freqs_list,
                    vibraciones=vibraciones,
                    cube_files=cube_files,
                    properties=props)

    except JobCancelled:
        _update_job(job_id, status="cancelled", message="Cancelado")
    except Exception as e:
        _update_job(job_id, status="error", progress=100, message=f"Error PySCF: {str(e)}")


# --------------------------------------------------------------------------- #
# Motor ORCA
# --------------------------------------------------------------------------- #
def ejecutar_orca(job_id, xyz_text, charge, multiplicity, basis, functional, task, cpu_cores=2):
    """
    Ejecuta un cálculo DFT con ORCA (proceso externo).

    Flujo:
        1. Construir geometry.inp a partir del XYZ.
        2. Lanzar el binario ORCA como subproceso (guardando el PID en
           ACTIVE_PROCESSES para poder cancelar).
        3. Monitorizar el output en tiempo real (tail de las últimas líneas)
           para actualizar el progreso (SCF, Opt, Freq, finalización/error).
        4. Al terminar, extraer la geometría final optimizada.
        5. Repetir SCF/Hessiano con PySCF (si está disponible) para obtener
           propiedades y modos de vibración coherentes con el resto del
           sistema.

    Parámetros:
        mismos que ``ejecutar_pyscf``.

    Además de ``status``/``progress``/``message``, deja en JOBS los campos
    ``geometry``, ``freqs``, ``vibraciones``, ``cube_files`` y ``properties``.
    """
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    input_path = job_dir / "geometry.inp"
    output_path = job_dir / "geometry.out"

    try:
        check_job_cancelled(job_id)
        _update_job(job_id, status="running", progress=10,
                    message=f"Preparando ORCA ({cpu_cores} núcleos)...")

        # ------------------ Construir el input de ORCA -------------------- #
        lines = xyz_text.strip().splitlines()
        atom_block = "\n".join(lines[2:])

        # Intento de obtener la base en formato ORCA desde BSE
        ext_basis_block = None
        if BSE_AVAILABLE:
            try:
                ext_basis_block = bse.get_basis(basis, fmt='orca')
            except Exception:
                pass
        use_ext = ext_basis_block is not None

        # Palabras clave: funcional + base (+ Opt/Freq según task)
        keywords = [functional]
        if not use_ext:
            keywords.append(basis)
        if task in ("opt", "opt_freq"):
            keywords.append("Opt")
        if task in ("freq", "opt_freq"):
            keywords.append("Freq")

        input_content = f"%pal\nnprocs {cpu_cores}\nend\n! {' '.join(keywords)}\n"
        if use_ext:
            input_content += f"%basis\n{ext_basis_block}\nend\n"
        input_content += f"* xyz {charge} {multiplicity}\n{atom_block}\n*\n"
        input_path.write_text(input_content)

        # ------------------ Lanzar el subproceso -------------------------- #
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = f"{ORCA_LIB_DIR}:{env.get('LD_LIBRARY_PATH', '')}"

        _update_job(job_id, progress=30, message="Ejecutando ORCA...")

        with open(output_path, "w") as out_f:
            process = subprocess.Popen(
                [ORCA_BINARY, input_path.name],
                cwd=job_dir, env=env,
                stdout=out_f, stderr=subprocess.STDOUT, text=True,
            )
        ACTIVE_PROCESSES[job_id] = process

        # ------------------ Monitorización en tiempo real ----------------- #
        start_time = time.time()
        while process.poll() is None:
            check_job_cancelled(job_id)

            # Timeout duro de 1 hora
            if time.time() - start_time > 3600:
                process.kill()
                raise TimeoutError("Timeout 1h")

            try:
                if output_path.exists():
                    with open(output_path, 'r', errors='ignore') as f:
                        # Leer solo los últimos ~5 KB del output
                        f.seek(0, os.SEEK_END)
                        f.seek(max(0, f.tell() - 5000))
                        tail_content = f.read()

                        # Detección de errores fatales
                        if ("aborting the run" in tail_content
                                or "ORCA finished by error termination" in tail_content
                                or "Error :" in tail_content):
                            process.kill()
                            error_lines = [
                                l.strip() for l in tail_content.split('\n')
                                if "Error" in l or "aborting" in l
                            ]
                            err_msg = error_lines[-1] if error_lines else "Error fatal en ORCA"
                            raise RuntimeError(f"ORCA abortó: {err_msg}")

                        # Actualización de progreso según la fase detectada
                        if "ORCA TERMINATED NORMALLY" in tail_content:
                            _update_job(job_id, progress=95, message="ORCA: Finalizando...")
                        elif "FREQUENCY" in tail_content and "Calculating" in tail_content:
                            _update_job(job_id, progress=80, message="ORCA: Calculando frecuencias...")
                        elif "GEOMETRY OPTIMIZATION CYCLE" in tail_content:
                            cycles = re.findall(r"GEOMETRY OPTIMIZATION CYCLE\s+(\d+)", tail_content)
                            cycle_num = cycles[-1] if cycles else "1"
                            _update_job(job_id, progress=60,
                                        message=f"ORCA: Optimizando geometría (Ciclo {cycle_num})...")
                        elif ("SCF ITERATIONS" in tail_content
                              or "Starting incremental Fock matrix" in tail_content):
                            _update_job(job_id, progress=40, message="ORCA: Iteraciones SCF...")
            except Exception as inner_e:
                # Si la excepción es la del abort de ORCA, propagar; si es
                # cualquier otra (lectura del fichero, etc.), ignorar.
                if "ORCA abortó" in str(inner_e):
                    raise inner_e
                pass
            time.sleep(1)

        # Liberar el PID activo
        if job_id in ACTIVE_PROCESSES:
            del ACTIVE_PROCESSES[job_id]

        if process.returncode != 0:
            err_msg = output_path.read_text(errors="replace")[-1000:]
            raise RuntimeError(f"ORCA falló con código {process.returncode}. Detalle:\n{err_msg}")

        # ------------------ Extraer geometría final ----------------------- #
        check_job_cancelled(job_id)
        _update_job(job_id, progress=95, message="Extrayendo resultados...")
        out_text = output_path.read_text(errors="replace")

        # Último bloque de coordenadas cartesianas de ORCA (geometría optimizada)
        gb = re.findall(
            r"CARTESIAN COORDINATES \(ANGSTROEM\)\s*-+\s*(.*?)(?=\n\s*-{10,})",
            out_text, re.DOTALL,
        )
        xyz_opt = xyz_text
        if gb:
            opt_lines = [l for l in gb[-1].strip().splitlines() if l.strip()]
            xyz_opt_lines = [str(len(opt_lines)), "Optimized"]
            for l in opt_lines:
                p = l.split()
                if len(p) >= 4:
                    xyz_opt_lines.append(
                        f"{p[0]:2s} {float(p[1]):.6f} {float(p[2]):.6f} {float(p[3]):.6f}"
                    )
            xyz_opt = "\n".join(xyz_opt_lines)

        # ------------------ Post-análisis con PySCF ----------------------- #
        vibraciones, freqs_list, cube_files, props = [], [], {}, {}
        if PYSCF_AVAILABLE:
            _update_job(job_id, progress=98, message="Análisis post-ORCA con PySCF...")
            try:
                # Base en formato PySCF (fallback al nombre original)
                pyscf_basis = basis
                if BSE_AVAILABLE:
                    try:
                        pyscf_basis = bse.get_basis(basis, fmt='pyscf')
                    except Exception:
                        pass

                mol = gto.M(
                    atom="; ".join(l.strip() for l in xyz_opt.strip().splitlines()[2:]),
                    basis=pyscf_basis, charge=charge, spin=multiplicity - 1, verbose=0,
                )
                mf = dft.RKS(mol) if functional != "HF" else scf.RHF(mol)
                if functional != "HF":
                    mf.xc = functional.lower()
                mf.kernel()

                # Propiedades electrónicas
                props["energy"] = f"{mf.e_tot:.6f}"
                props["converged"] = "Sí" if mf.converged else "No"
                mo_e = np.array(mf.mo_energy).flatten()
                mo_o = np.array(mf.mo_occ).flatten()
                h, l = np.where(mo_o > 0)[0], np.where(mo_o == 0)[0]
                if len(h) > 0 and len(l) > 0:
                    props["homo"] = f"{mo_e[max(h)] * 27.2114:.3f} eV"
                    props["lumo"] = f"{mo_e[min(l)] * 27.2114:.3f} eV"
                    props["gap"] = f"{(mo_e[min(l)] - mo_e[max(h)]) * 27.2114:.3f} eV"
                try:
                    dip = mf.dip_moment(unit='Debye')
                    props["dipole"] = f"{np.linalg.norm(dip):.3f} D"
                except Exception:
                    props["dipole"] = "N/A"

                # Cubos de orbitales moleculares
                for idx in range(mol.nao_nr()):
                    try:
                        cubegen.orbital(mol, str(job_dir / f"mo_{idx}.cube"), mf.mo_coeff[:, idx])
                        cube_files[str(idx)] = f"/jobs/{job_id}/mo_{idx}.cube"
                    except Exception:
                        pass

                # Frecuencias con PySCF (si task las pide)
                if task in ("freq", "opt_freq"):
                    hess = mf.Hessian().kernel()
                    res = thermo.harmonic_analysis(mol, hess)
                    freqs, modes = res['freq_wavenumber'], res['norm_mode']
                    freqs_list = freqs.tolist()
                    xyz_base = "\n".join(
                        [" ".join(l.split()[:4]) for l in xyz_opt.splitlines()[2:]]
                    )
                    n_atoms = len(xyz_base.strip().splitlines())
                    vibraciones = _build_vibrations(xyz_base, freqs, modes, n_atoms)
            except Exception as e:
                logger.error(f"Error PySCF post-ORCA: {e}")

        # ------------------ Final OK -------------------------------------- #
        check_job_cancelled(job_id)
        _update_job(job_id, status="done", progress=100, message="ORCA completado",
                    geometry=xyz_opt,
                    freqs=freqs_list,
                    vibraciones=vibraciones,
                    cube_files=cube_files,
                    properties=props)

    except JobCancelled:
        # Cancelación explícita: matar proceso si sigue vivo
        if job_id in ACTIVE_PROCESSES:
            try:
                ACTIVE_PROCESSES[job_id].kill()
            except Exception:
                pass
            del ACTIVE_PROCESSES[job_id]
        _update_job(job_id, status="cancelled", message="Cancelado")

    except Exception as e:
        # Cualquier otro error: matar proceso si sigue vivo y registrar
        if job_id in ACTIVE_PROCESSES:
            try:
                ACTIVE_PROCESSES[job_id].kill()
            except Exception:
                pass
            del ACTIVE_PROCESSES[job_id]
        _update_job(job_id, status="error", progress=100, message=f"Error DFT: {str(e)}")


# --------------------------------------------------------------------------- #
# Endpoint: previsualización desde SMILES
# --------------------------------------------------------------------------- #
@bp.route("/api/preview", methods=["POST"])
def preview():
    """
    Genera un bloque XYZ a partir de un SMILES para previsualización.

    Entrada JSON:
        { "smiles": "<cadena>" }

    Salida JSON:
        { "xyz": "<bloque XYZ>" }  o  { "error": "<msg>" }
    """
    data = request.get_json() or {}
    try:
        xyz = smiles_to_xyz(data.get("smiles", ""))
        return jsonify({"xyz": xyz})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ==============================================================================
# ENDPOINT ACTUALIZADO: carga .xyz / .inp / .pdb y devuelve SMILES
# ==============================================================================
@bp.route("/api/upload_structure", methods=["POST"])
def upload_structure():
    """
    Sube un fichero de estructura y devuelve un JSON normalizado.

    Formatos soportados:
        - .xyz  : se analiza directamente (fórmula, e-, SMILES).
        - .inp  : (ORCA/Gaussian) se convierte a XYZ y se analiza.
        - .pdb  : se devuelve el texto tal cual (para otros módulos).

    Salida JSON (según formato):
        - xyz: { type, text, meta, smiles, charge, multiplicity }
        - inp: { type, text, original_inp, meta, smiles, charge, multiplicity }
        - pdb: { type, text }
    """
    try:
        f = request.files.get("file")
        if not f:
            return jsonify({"error": "No hay archivo"}), 400

        content = f.read().decode("utf-8", errors="replace")
        ext = f.filename.lower().split('.')[-1]

        if ext == "xyz":
            meta = analyze_xyz(content)
            return jsonify({
                "type": "xyz",
                "text": content,
                "meta": meta,
                "smiles": meta.get("smiles"),
                "charge": 0,
                "multiplicity": 1,
            })

        elif ext == "inp":
            try:
                xyz_text, charge, mult = parse_inp_to_xyz(content)
            except Exception as e:
                return jsonify({"error": f"Error parseando .inp: {e}"}), 400
            meta = analyze_xyz(xyz_text, charge=charge)
            return jsonify({
                "type": "inp",
                "text": xyz_text,
                "original_inp": content,
                "meta": meta,
                "smiles": meta.get("smiles"),
                "charge": charge,
                "multiplicity": mult,
            })

        elif ext == "pdb":
            return jsonify({"type": "pdb", "text": content})

        return jsonify({"error": "Formato no soportado. Use .xyz, .inp o .pdb"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --------------------------------------------------------------------------- #
# Endpoint: lanzar cálculo DFT
# --------------------------------------------------------------------------- #
@bp.route("/api/calcular", methods=["POST"])
def calcular():
    """
    Lanza un cálculo DFT en un hilo en segundo plano.

    Entrada JSON (todos opcionales salvo alguno de xyz_text/smiles):
        {
            "functional":   "B3LYP",
            "basis":        "def2-SVP",
            "xyz_text":     "<bloque XYZ>",     # o bien...
            "smiles":       "<SMILES>",         # ...esto
            "cpu_cores":    2,
            "task":         "opt_freq" | "opt" | "freq" | "single",
            "charge":       0,
            "multiplicity": 1,
            "motor":        "orca" | "pyscf"
        }

    Salida JSON:
        { "job_id": "<uuid>", "status": "running" }  o  { "error": "<msg>" }

    El cálculo actualiza ``JOBS[job_id]`` y puede monitorizarse con
    ``/api/estado/<job_id>``.
    """
    data = request.json or {}
    functional = data.get("functional", "B3LYP")
    basis = data.get("basis", "def2-SVP")

    # Validación temprana de funcional + base
    valid, err = validate_methods(functional, basis)
    if not valid:
        return jsonify({"error": err}), 400

    try:
        # Geometría: preferir xyz_text; si no, generar desde SMILES
        xyz = data.get("xyz_text") or smiles_to_xyz(data.get("smiles", ""))

        # Identificador único del trabajo
        job_id = str(uuid.uuid4())
        cpu_cores = int(data.get("cpu_cores", 2))

        # Registro inicial del job en el diccionario global
        _update_job(
            job_id,
            id=job_id,
            status="running",
            progress=0,
            start_time=time.time(),
            geometry=xyz,
            task=data.get("task", "opt_freq"),
            charge=int(data.get("charge", 0)),
            multiplicity=int(data.get("multiplicity", 1)),
            basis=basis,
            functional=functional,
            engine=data.get("motor", "orca"),
        )

        # Selección del motor: ORCA (por defecto) o PySCF
        engine = ejecutar_orca if data.get("motor", "orca") == "orca" else ejecutar_pyscf
        args = (
            job_id, xyz,
            int(data.get("charge", 0)),
            int(data.get("multiplicity", 1)),
            basis, functional,
            data.get("task", "opt_freq"),
            cpu_cores,
        )

        # Lanzar en hilo separado (no bloquea la petición HTTP)
        threading.Thread(target=engine, args=args).start()

        return jsonify({"job_id": job_id, "status": "running"})

    except Exception as e:
        return jsonify({"error": str(e)}), 400
