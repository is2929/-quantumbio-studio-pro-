# QuantumBio Studio Pro — versión modular

La aplicación se organiza en dos módulos científicos conectados desde `main.py`.

## Estructura

```text
QuantumBioStudio_Modular_v2/
├── main.py                         # Aplicación Flask y registro de módulos
├── core.py                         # Estado compartido, dependencias y utilidades comunes
├── modules/
│   ├── __init__.py
│   ├── quimica_computacional.py    # API y motores PySCF/ORCA
│   └── bioinformatica.py           # PDB, docking, MD, AlphaFold y PyMOL
├── templates/
│   └── index.html                  # Interfaz web conectada con las API
├── static/
│   └── uploads/
├── jobs/                           # Se crea/usa para los trabajos
├── models/                         # Se crea/usa para modelos
└── requirements.txt
```

## Ejecutar

Desde esta carpeta, crea un entorno virtual e instala las dependencias:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

Después abre `http://127.0.0.1:5001`.

Si ORCA está instalado en otra ubicación, configura la ruta antes de ejecutar:

```bash
export ORCA_DIR="/ruta/a/orca"
python main.py
```

## Módulos y conexión

`main.py` crea la aplicación Flask e importa y registra los blueprints:

- `modules.quimica_computacional.bp`
- `modules.bioinformatica.bp`

`core.py` centraliza el estado compartido de los trabajos, sus bloqueos, las carpetas de resultados y las utilidades que necesitan ambos módulos. Así ambos módulos consultan el mismo estado para que la ruta `/api/estado/<job_id>` pueda devolver el progreso de cualquier cálculo.

## Endpoints conservados

### Química computacional
- `GET /api/list_basis_sets`
- `POST /api/preview`
- `POST /api/upload_structure`
- `POST /api/calcular`

### Bioinformática
- `POST /api/prepare_pdb`
- `GET /api/fetch_pdb/<pdb_id>`
- `POST /api/docking`
- `POST /api/md`
- `POST /api/md_from_docking`
- `POST /api/alphafold`
- `POST /api/export_to_pymol`
- `POST /api/pymol_command`

### Rutas comunes
- `GET /`
- `GET /api/system_info`
- `POST /api/ai_assistant`
- `POST /api/cancelar/<job_id>`
- `GET /api/estado/<job_id>`
- `POST /api/generate_report`
- `GET /jobs/<job_id>/<filename>`

## Dependencias externas

Instalar `requirements.txt` no instala necesariamente todos los ejecutables externos. Según la función usada, también pueden necesitarse:

- ORCA para cálculos químicos con ese motor.
- AutoDock Vina (`vina`) y Open Babel (`obabel`) para docking.
- PyMOL en modo servidor (`pymol -R`) para las operaciones XML-RPC.
- OpenMM, PDBFixer, OpenMM ForceFields y OpenFF para dinámica molecular.

La aplicación puede iniciar aunque algunas bibliotecas científicas opcionales no estén disponibles, pero la función que dependa de ellas no se podrá ejecutar hasta instalarlas.

## Nota sobre la interfaz

El archivo fuente recibido en esta entrega era el código Python. No incluía el contenido del `templates/index.html` original. Por eso este ZIP incorpora una interfaz web funcional que consume las rutas conservadas; no pretende ser una copia exacta del diseño HTML original.
