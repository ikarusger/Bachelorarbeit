from dataclasses import dataclass
from typing import Dict, Tuple
from pathlib import Path

import numpy as np
import trimesh
from itertools import permutations, product


Vec3 = Tuple[float, float, float]  # (x, y, z)


@dataclass(frozen=True)
class Duese:
    name: str
    pos: Vec3
    kraft: str  # "u" = unten, "s" = seite


# Koordinatensystem-Konvention:
# x = vorne, y = links, z = oben
Ursprung: Vec3 = (0.0, 0.0, 0.0)

# Duesen "Datenbank"
# Zugriff z.B.: print(duesen["DueseU1"].pos[0])
duesen: Dict[str, Duese] = {   #   x    y    z
    "DueseU1": Duese("DueseU1", (10.0, 10.0, 0.0), "u"),
    "DueseU2": Duese("DueseU2", (10.0, 20.0, 0.0), "u"),
    "DueseU3": Duese("DueseU3", (10.0, 30.0, 0.0), "u"),
    "DueseU4": Duese("DueseU4", (20.0, 10.0, 0.0), "u"),
    "DueseU5": Duese("DueseU5", (20.0, 20.0, 0.0), "u"),
    "DueseU6": Duese("DueseU6", (20.0, 30.0, 0.0), "u"),
    "DueseU7": Duese("DueseU7", (30.0, 10.0, 0.0), "u"),
    "DueseU8": Duese("DueseU8", (30.0, 20.0, 0.0), "u"),

    "DueseO1": Duese("DueseO1", (0.0, 10.0, 10.0), "s"),
    "DueseO2": Duese("DueseO2", (0.0, 20.0, 10.0), "s"),
    "DueseO3": Duese("DueseO3", (0.0, 30.0, 10.0), "s"),
    "DueseO4": Duese("DueseO4", (0.0, 10.0, 20.0), "s"),
    "DueseO5": Duese("DueseO5", (0.0, 20.0, 20.0), "s"),
    "DueseO6": Duese("DueseO6", (0.0, 30.0, 20.0), "s"),
    "DueseO7": Duese("DueseO7", (0.0, 10.0, 30.0), "s"),
    "DueseO8": Duese("DueseO8", (0.0, 20.0, 30.0), "s"),
}


def load_mesh(path: str) -> trimesh.Trimesh:
    """
    Lädt eine Mesh-Datei (STL/OBJ/PLY).
    Falls trimesh eine Scene lädt (mehrere Teile), werden sie zu einem Mesh zusammengefügt.
    """
    obj = trimesh.load(path, force="mesh")

    if isinstance(obj, trimesh.Scene):
        obj = trimesh.util.concatenate(tuple(obj.geometry.values()))
        print("Hinweis: Datei wurde als Scene geladen -> Meshes wurden zusammengefügt.")

    if not isinstance(obj, trimesh.Trimesh):
        raise TypeError(f"Datei konnte nicht als Mesh geladen werden: {path}")

    return obj



def rot_z_90k(k: int) -> np.ndarray:
    """Drehung um z-Achse: k * 90° als 3x3 Matrix."""
    k = k % 4 # stellt sicher, dass auch wenn Werte über 4 eingeben werden, es funktioniert. Dies benötige ich, wenn ich zum Bsp. Drehungen addiere.
    if k == 0:
        return np.eye(3)   # Einheitsmatrix wird erstellt 1, 0, 0    0, 1, 0   0, 0, 1
    if k == 1:  # 90°
        return np.array([[0, -1, 0],
                         [1,  0, 0],
                         [0,  0, 1]], dtype=float)
    if k == 2:  # 180°
        return np.array([[-1, 0, 0],
                         [ 0, -1, 0],
                         [ 0,  0, 1]], dtype=float)
    # k == 3: 270°
    return np.array([[ 0, 1, 0],
                     [-1, 0, 0],
                     [ 0, 0, 1]], dtype=float)

def rot_x_90k(k: int) -> np.ndarray:
    k = k % 4
    if k == 0:
        return np.eye(3)
    if k == 1:  # 90°
        return np.array([[1, 0, 0],
                         [0, 0, -1],
                         [0, 1, 0]], dtype=float)
    if k == 2:  # 180°
        return np.array([[1, 0, 0],
                         [0, -1, 0],
                         [0, 0, -1]], dtype=float)
    # 270°
    return np.array([[1, 0, 0],
                     [0, 0, 1],
                     [0, -1, 0]], dtype=float)


def rot_y_90k(k: int) -> np.ndarray:
    k = k % 4
    if k == 0:
        return np.eye(3)
    if k == 1:  # 90°
        return np.array([[0, 0, 1],
                         [0, 1, 0],
                         [-1, 0, 0]], dtype=float)
    if k == 2:  # 180°
        return np.array([[-1, 0, 0],
                         [0, 1, 0],
                         [0, 0, -1]], dtype=float)
    # 270°
    return np.array([[0, 0, -1],
                     [0, 1, 0],
                     [1, 0, 0]], dtype=float)

def positionieren24() -> list[tuple[np.ndarray, str]]:
    """
    Liefert 24 Rotationen in einer einfachen, logischen Reihenfolge:
    - Position 1..4: nur z-Drehungen (0/90/180/270)
    - Position 5..8: nächste "Grundlage" + wieder 0/90/180/270 um z
    - usw.
    """
    I = np.eye(3)  # Einheitsmatrix wird erstellt

    # 6 Grundlagen = welche Seite "unten" ist (vereinfachtes, gut verständliches Set)
    bases = [    # alle 6 Flächen liegen einmal unten
        ("x0", I),              # Basis 1: wie geladen
        ("x180", rot_x_90k(2)), # Basis 2: auf den Kopf (180° um x)
        ("y90", rot_y_90k(1)),  # Basis 3: um y kippen
        ("y270", rot_y_90k(3)), # Basis 4: um y anders kippen
        ("x90", rot_x_90k(1)),  # Basis 5: um x kippen
        ("x270", rot_x_90k(3)), # Basis 6: um x anders kippen
    ]

    positions = []
    for base_label, base in bases:
        for k in range(4):
            # erst "Basis" (Kippen), dann Drehung um die Welt-z-Achse
            z_deg = k * 90
            label = f"{base_label}_z{z_deg}"
            positions.append((rot_z_90k(k) @ base, label))

    return positions

def build_origin_marker(origin: Vec3, size_ref: float) -> trimesh.Trimesh:
    parts = []
    base = max(2.0, size_ref * 0.05)
    origin_radius = base
    # Ursprung als Kugel + Achskreuz
    parts.append(trimesh.creation.icosphere(subdivisions=2, radius=origin_radius, center=origin))
    parts.append(trimesh.creation.box(extents=(base * 3, base * 0.6, base * 0.6), transform=trimesh.transformations.translation_matrix(origin)))
    parts.append(trimesh.creation.box(extents=(base * 0.6, base * 3, base * 0.6), transform=trimesh.transformations.translation_matrix(origin)))
    parts.append(trimesh.creation.box(extents=(base * 0.6, base * 0.6, base * 3), transform=trimesh.transformations.translation_matrix(origin)))
    return trimesh.util.concatenate(parts)

def build_nozzles_mesh(nozzle_map: Dict[str, Duese], size_ref: float) -> trimesh.Trimesh:
    parts = []
    base = max(2.0, size_ref * 0.05)
    radius = base
    height = base * 2.0
    for noz in nozzle_map.values():
        if noz.kraft == "u":
            axis = np.array([0.0, 0.0, -1.0])  # unten
        else:
            axis = np.array([1.0, 0.0, 0.0])   # seite -> nach vorne (+x)
        center = np.array(noz.pos, dtype=float) + axis * (height * 0.5)
        cyl = trimesh.creation.cylinder(radius=radius, height=height, sections=32)
        align = trimesh.geometry.align_vectors([0, 0, 1], axis)
        if align is None:
            align = np.eye(4)
        cyl.apply_transform(align)
        cyl.apply_translation(center)
        parts.append(cyl)
    return trimesh.util.concatenate(parts)

if __name__ == "__main__":
    #  Pfad zu  STL-Datei  
    pfad_teil1 = Path(r"C:\Users\micha\Desktop\Bachelorarbeit\Programmierung\Cad Modelle\Test-Bauteil1.stl")
    print("STL-Pfad:", pfad_teil1)
    print("Existiert die Datei?", pfad_teil1.exists())

    if not pfad_teil1.exists():
        raise FileNotFoundError(f"STL-Datei nicht gefunden: {pfad_teil1}")

    # Mesh laden und ein paar Infos ausgeben
    m = load_mesh(str(pfad_teil1))
    print("Mesh geladen!")


    rotations = positionieren24()   # Liste mit 24 Matrizen (3x3)
    size_ref = float((m.bounds[1] - m.bounds[0]).max())
    origin_marker = build_origin_marker(Ursprung, size_ref)
    nozzles_mesh = build_nozzles_mesh(duesen, size_ref)




    out_dir = Path(r"C:\Users\micha\Desktop\Bachelorarbeit\Programmierung\Cad Modelle\Modelle1")
    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = pfad_teil1.stem

    for position_id, (R3, label) in enumerate(rotations, start=1):
        T = np.eye(4)        # 4x4 Einheitsmatrix
        T[:3, :3] = R3       # oben links die 3x3 Rotation einsetzen

        m_rot = m.copy()     # Kopie, Original bleibt unverändert
        m_rot.apply_transform(T)
        # Keine zusätzliche Verschiebung: Original-Koordinaten beibehalten.
        export_mesh = trimesh.util.concatenate([m_rot, origin_marker, nozzles_mesh])

        out_path = out_dir / f"{base_name}_pos{position_id:02d}_{label}.stl"
        export_mesh.export(out_path)
        print("Exportiert:", out_path)

        if position_id == 1 and label == "x0_z0":
            bmin, bmax = m_rot.bounds
            corners = [
                (bmin[0], bmin[1], bmin[2]),
                (bmin[0], bmin[1], bmax[2]),
                (bmin[0], bmax[1], bmin[2]),
                (bmin[0], bmax[1], bmax[2]),
                (bmax[0], bmin[1], bmin[2]),
                (bmax[0], bmin[1], bmax[2]),
                (bmax[0], bmax[1], bmin[2]),
                (bmax[0], bmax[1], bmax[2]),
            ]
            print("Ecken Test-Bauteil1_pos01_x0_z0:")
            for i, c in enumerate(corners, start=1):
                print(f"  Ecke {i}: {c}")
            print("Ursprung:", (0.0, 0.0, 0.0))
            print("Duesen (fixes Koordinatensystem):")
            for name, d in duesen.items():
                print(f"  {name}: {d.pos}")

    # Anzeige entfernt; Export der Dateien reicht.

    
