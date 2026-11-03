"""Estado compartido y dependencias comunes de QuantumBio Studio."""

import os

import re

import uuid

import json

import shutil

import time

import math

import psutil

import logging

import threading

import subprocess

import xmlrpc.client

from pathlib import Path

from io import StringIO

from datetime import datetime

from flask import Flask, render_template, request, jsonify, send_from_directory

from werkzeug.exceptions import RequestEntityTooLarge

from apscheduler.schedulers.background import BackgroundScheduler

import numpy as np

import requests

from rdkit import Chem

from rdkit.Chem import AllChem, rdDetermineBonds, Descriptors, Crippen, Lipinski

try:
    import basis_set_exchange as bse
    BSE_AVAILABLE = True
except ImportError:
    BSE_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from pyscf import gto, scf, dft, hessian
    from pyscf.hessian import thermo
    from pyscf.tools import cubegen
    PYSCF_AVAILABLE = True
except ImportError:
    PYSCF_AVAILABLE = False

try:
    from Bio.PDB import PDBParser
    BIO_AVAILABLE = True
except ImportError:
    BIO_AVAILABLE = False

try:
    import openmm as mm
    import openmm.app as omm_app
    import openmm.unit as unit
    OPENMM_AVAILABLE = True
except ImportError:
    OPENMM_AVAILABLE = False

try:
    from pdbfixer import PDBFixer
    PDBFIXER_AVAILABLE = True
except ImportError:
    PDBFIXER_AVAILABLE = False

try:
    from openmmforcefields.generators import SystemGenerator
    from openff.toolkit.topology import Molecule as OFFMolecule
    OPENMMFF_AVAILABLE = True
except ImportError:
    OPENMMFF_AVAILABLE = False
    logging.warning("openmmforcefields/openff-toolkit no instalados.")

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle, Paragraph,
                                    Spacer, Image as RLImage, PageBreak, KeepTogether)
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, cm
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "static" / "uploads"

JOBS_DIR = BASE_DIR / "jobs"

MODELS_DIR = BASE_DIR / "models"

BSE_CACHE_FILE = BASE_DIR / "bse_cache.json"

for d in (UPLOAD_DIR, JOBS_DIR, MODELS_DIR): d.mkdir(parents=True, exist_ok=True)

ORCA_DIR = os.environ.get("ORCA_DIR", "/home/isaac/Documentos/orca_caperta/orca_5_0_1_linux_x86-64_shared_openmpi411")

ORCA_BINARY = f"{ORCA_DIR}/orca"

ORCA_LIB_DIR = ORCA_DIR

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

JOBS: dict[str, dict] = {}

JOBS_LOCK = threading.Lock()

ACTIVE_PROCESSES: dict[str, subprocess.Popen] = {}

scheduler = BackgroundScheduler()

FUNCTIONALS = {
    'BLYP', 'BP86', 'PBE', 'REVPBE', 'RPBE', 'SCANFUNC', 'RSCAN', 'R2SCAN', 'TPSS', 'B97M-V',
    'B3LYP', 'B3P86', 'B3PW91', 'MPW1PW', 'PBE0', 'PW1PW', 'PW6B95', 'REVPBE0', 'REVPBE38',
    'TPSSH', 'TPSS0', 'R2SCANH', 'R2SCAN0', 'R2SCAN50', 'M06-2X', 'WB97X-D', 'CAM-B3LYP',
    'B97-3C', 'WB97X-3C', 'R2SCAN-3C', 'HF'
}

NATIVE_ORCA_BASES = {
    "DEF2-SVP", "DEF2-TZVP", "DEF2-TZVPP", "DEF2-QZVP", "DEF2-SV(P)", "DEF2-SVPD", "DEF2-TZVPD",
    "6-31G", "6-31G*", "6-31G**", "6-31+G*", "6-311G**", "6-311++G**",
    "CC-PVDZ", "CC-PVTZ", "CC-PVQZ", "AUG-CC-PVDZ", "AUG-CC-PVTZ", "STO-3G", "3-21G"
}

NATIVE_PYSCF_BASES = {
    "DEF2-SVP", "DEF2-TZVP", "DEF2-TZVPP", "DEF2-QZVP", "6-31G", "6-31G*", "6-31G**", "6-311G**", "CC-PVDZ", "CC-PVTZ", "STO-3G", "6-31G(D)", "6-31G(D,P)", "6-311G(D,P)"
}

class JobCancelled(Exception): pass

def check_job_cancelled(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job and job.get("status") == "cancelled": raise JobCancelled("Trabajo cancelado")

def cleanup_old_jobs():
    now = time.time()
    with JOBS_LOCK:
        to_delete = [jid for jid, job in JOBS.items() if now - job.get("start_time", now) > 86400]
        for jid in to_delete:
            del JOBS[jid]
            shutil.rmtree(JOBS_DIR / jid, ignore_errors=True)

scheduler.add_job(func=cleanup_old_jobs, trigger="interval", hours=1)

scheduler.start()

def _update_job(job_id, **kw):
    with JOBS_LOCK:
        job = JOBS.setdefault(job_id, {})
        if "start_time" not in job: job["start_time"] = time.time()
        job.update(kw)
        if job.get("status") == "running" and job.get("progress", 0) > 0:
            elapsed = time.time() - job["start_time"]
            total = elapsed / (job["progress"] / 100.0)
            job["eta"] = int(max(0, total - elapsed))
            job["elapsed"] = int(elapsed)

def _split_models_from_file(path):
    poses, cur = [], []
    with open(path, errors="ignore") as f:
        for line in f:
            if line.startswith("MODEL"):
                if cur: poses.append("".join(cur))
                cur = []
            elif line.startswith("ENDMDL"):
                continue
            else:
                cur.append(line)
    if cur: poses.append("".join(cur))
    return poses

def calc_rmsd(P, Q):
    P_mean = P.mean(axis=0); Q_mean = Q.mean(axis=0)
    P_c = P - P_mean; Q_c = Q - Q_mean
    C = np.dot(P_c.T, Q_c)
    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0
    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]
    U = np.dot(V, W)
    P_rot = np.dot(P_c, U)
    return np.sqrt(((P_rot - Q_c) ** 2).sum() / P.shape[0])

def calc_rg(P):
    com = P.mean(axis=0)
    return np.sqrt(((P - com) ** 2).sum(axis=1).mean())

ATOMIC_Z = {"H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Br": 35, "I": 53}

def calcular_ki(score, temp=298.15):
    R = 1.987e-3
    return float(np.exp(score / (R * temp)))
    
    
# ==============================================================================
# ANIMACIÓN DE MODOS NORMALES (vibraciones)
# ==============================================================================
def _extract_mode_vector(modes, idx, n_freqs, n_atoms):
    if modes is None or not hasattr(modes, "ndim") or modes.ndim != 3:
        return None
    if modes.shape[0] == n_freqs:
        m = modes[idx]
    elif modes.shape[-1] == n_freqs:
        m = modes[:, :, idx]
    elif modes.shape[1] == n_freqs:
        m = modes[:, idx, :]
    else:
        return None

    m = np.asarray(m, dtype=float)
    if m.shape == (3, n_atoms) and n_atoms != 3:
        m = m.T
    if m.shape != (n_atoms, 3):
        return None
    return m


def _generate_mode_animation(xyz_base, mode_vec, n_atoms, frames=20, amplitude=0.5):
    lines = [l.split() for l in xyz_base.strip().splitlines() if l.strip()]
    if len(lines) != n_atoms:
        return ""
    atoms = [l[0] for l in lines]
    coords = np.array([[float(x) for x in l[1:4]] for l in lines])

    mode_vec = np.asarray(mode_vec, dtype=float)
    if mode_vec.shape != (n_atoms, 3):
        return ""

    out = []
    for f in range(frames):
        t = (f / frames) * 2.0 * np.pi
        disp = np.sin(t) * amplitude * mode_vec
        fc = coords + disp
        out.append(f"{n_atoms}\nFrame {f}")
        for i, a in enumerate(atoms):
            out.append(f"{a} {fc[i,0]:.6f} {fc[i,1]:.6f} {fc[i,2]:.6f}")
    return "\n".join(out) + "\n"


def _build_vibrations(xyz_base, freqs, modes, n_atoms, min_freq=10.0):
    vibraciones = []
    n_freqs = len(freqs)
    for idx, f in enumerate(freqs):
        if f < min_freq:
            continue
        mode_vec = _extract_mode_vector(modes, idx, n_freqs, n_atoms)
        if mode_vec is None:
            continue
        xyz_anim = _generate_mode_animation(xyz_base, mode_vec, n_atoms)
        if xyz_anim:
            vibraciones.append({
                "index": int(idx),
                "frecuencia": float(f),
                "xyz_anim": xyz_anim,
            })
    return vibraciones
