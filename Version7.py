from dataclasses import dataclass
from typing import Dict, Tuple
from pathlib import Path

import numpy as np
import trimesh
from itertools import permutations, product


Vec3 = Tuple[float, float, float]  # (x, y, z)

CylinderRadius = 2.0
CylinderHeight = 4.0


@dataclass(frozen=True)
class Düse:
    name: str
    pos: Vec3
    kraft: str  # "u" = unten, "s" = seite


# Koordinatensystem-Konvention:
# x = vorne, y = links, z = oben
Ursprung: Vec3 = (0.0, 0.0, 0.0)

# Düsen "Datenbank"
# Zugriff z.B.: print(düsen["DüseU1"].pos[0])
düsen: Dict[str, Düse] = {   #   x    y    z
    "DüseU1": Düse("DüseU1", (10.0, 10.0, 0.0), "u"),
    "DüseU2": Düse("DüseU2", (10.0, 20.0, 0.0), "u"),
    "DüseU3": Düse("DüseU3", (10.0, 30.0, 0.0), "u"),
    "DüseU4": Düse("DüseU4", (20.0, 10.0, 0.0), "u"),
    "DüseU5": Düse("DüseU5", (20.0, 20.0, 0.0), "u"),
    "DüseU6": Düse("DüseU6", (20.0, 30.0, 0.0), "u"),
    "DüseU7": Düse("DüseU7", (30.0, 10.0, 0.0), "u"),
    "DüseU8": Düse("DüseU8", (30.0, 20.0, 0.0), "u"),

    "DüseO1": Düse("DüseO1", (0.0, 10.0, 10.0), "s"),
    "DüseO2": Düse("DüseO2", (0.0, 20.0, 10.0), "s"),
    "DüseO3": Düse("DüseO3", (0.0, 30.0, 10.0), "s"),
    "DüseO4": Düse("DüseO4", (0.0, 10.0, 20.0), "s"),
    "DüseO5": Düse("DüseO5", (0.0, 20.0, 20.0), "s"),
    "DüseO6": Düse("DüseO6", (0.0, 30.0, 20.0), "s"),
    "DüseO7": Düse("DüseO7", (0.0, 10.0, 30.0), "s"),
    "DüseO8": Düse("DüseO8", (0.0, 20.0, 30.0), "s"),
}


def load_mesh(path: str) -> trimesh.Trimesh:
    """
    LÃ¤dt eine Mesh-Datei (STL/OBJ/PLY).
    Falls trimesh eine Scene lÃ¤dt (mehrere Teile), werden sie zu einem Mesh zusammengefügt.
    """
    obj = trimesh.load(path, force="mesh")

    if isinstance(obj, trimesh.Scene):
        obj = trimesh.util.concatenate(tuple(obj.geometry.values()))
        print("Hinweis: Datei wurde als Scene geladen -> Meshes wurden zusammengefügt.")

    if not isinstance(obj, trimesh.Trimesh):
        raise TypeError(f"Datei konnte nicht als Mesh geladen werden: {path}")

    return obj


def rot_z_90k(k: int) -> np.ndarray:
    """Drehung um z-Achse: k * 90Â° als 3x3 Matrix."""
    k = k % 4 # stellt sicher, dass auch wenn Werte über 4 eingeben werden, es funktioniert. Dies benötige ich, wenn ich zum Bsp. Drehungen addiere.
    if k == 0:
        return np.eye(3)   # Einheitsmatrix wird erstellt 1, 0, 0    0, 1, 0   0, 0, 1
    if k == 1:  # 90Â°
        return np.array([[0, -1, 0],
                         [1,  0, 0],
                         [0,  0, 1]], dtype=float)
    if k == 2:  # 180Â°
        return np.array([[-1, 0, 0],
                         [ 0, -1, 0],
                         [ 0,  0, 1]], dtype=float)
    # k == 3: 270Â°
    return np.array([[ 0, 1, 0],
                     [-1, 0, 0],
                     [ 0, 0, 1]], dtype=float)

def rot_x_90k(k: int) -> np.ndarray:
    k = k % 4
    if k == 0:
        return np.eye(3)
    if k == 1:  # 90Â°
        return np.array([[1, 0, 0],
                         [0, 0, -1],
                         [0, 1, 0]], dtype=float)
    if k == 2:  # 180Â°
        return np.array([[1, 0, 0],
                         [0, -1, 0],
                         [0, 0, -1]], dtype=float)
    # 270Â°
    return np.array([[1, 0, 0],
                     [0, 0, 1],
                     [0, -1, 0]], dtype=float)


def rot_y_90k(k: int) -> np.ndarray:
    k = k % 4
    if k == 0:
        return np.eye(3)
    if k == 1:  # 90Â°
        return np.array([[0, 0, 1],
                         [0, 1, 0],
                         [-1, 0, 0]], dtype=float)
    if k == 2:  # 180Â°
        return np.array([[-1, 0, 0],
                         [0, 1, 0],
                         [0, 0, -1]], dtype=float)
    # 270Â°
    return np.array([[0, 0, -1],
                     [0, 1, 0],
                     [1, 0, 0]], dtype=float)

def positionieren24() -> list[tuple[np.ndarray, str]]:
    """
    Liefert 24 Rotationen in einer einfachen, logischen Reihenfolge:
    - Position 1..4: nur z-Drehungen (0/90/180/270)
    - Position 5..8: nÃ¤chste "Grundlage" + wieder 0/90/180/270 um z
    - usw.
    """
    I = np.eye(3)  # Einheitsmatrix wird erstellt

    # 6 Grundlagen = welche Seite "unten" ist (vereinfachtes, gut verstÃ¤ndliches Set)
    bases = [    # alle 6 FlÃ¤chen liegen einmal unten
        ("x0", I),              # Basis 1: wie geladen
        ("x180", rot_x_90k(2)), # Basis 2: auf den Kopf (180Â° um x)
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



def ecken_und_kanten_aus_bounds(bounds: np.ndarray) -> dict:
    minx, miny, minz = bounds[0]
    maxx, maxy, maxz = bounds[1]
    ecken = [
        (minx, miny, maxz),  # E1
        (minx, miny, minz),  # E2
        (minx, maxy, maxz),  # E3
        (minx, maxy, minz),  # E4
        (maxx, miny, maxz),  # E5
        (maxx, miny, minz),  # E6
        (maxx, maxy, maxz),  # E7
        (maxx, maxy, minz),  # E8
    ]
    kanten = [
        (1, 2), (1, 3), (1, 5),
        (2, 4), (2, 6),
        (3, 4), (3, 7),
        (4, 8),
        (5, 6), (5, 7),
        (6, 8),
        (7, 8),
    ]
    kanten_coords = [
        {"edge": (a, b), "from": ecken[a - 1], "to": ecken[b - 1]}
        for (a, b) in kanten
    ]
    return {"ecken": ecken, "kanten": kanten, "kanten_coords": kanten_coords}

def create_nozzle_cylinders(positions: list[np.ndarray]) -> trimesh.Trimesh | None:
    if not positions:
        return None
    cylinders = []
    for pos in positions:
        cyl = trimesh.creation.cylinder(
            radius=CylinderRadius,
            height=CylinderHeight,
            sections=16,
        )
        cyl.apply_translation([0.0, 0.0, CylinderHeight / 2.0])
        cyl.apply_translation(pos)
        cylinders.append(cyl)
    return trimesh.util.concatenate(cylinders)

if __name__ == "__main__":
    #  Pfad zu  STL-Datei  
    pfad_teil1 = Path(r"C:\Users\micha\Desktop\Bachelorarbeit\Programmierung\Cad Modelle\Test-Bauteil1.stl")
    print("STL-Pfad:", pfad_teil1)
    print("Existiert die Datei?", pfad_teil1.exists())

    if not pfad_teil1.exists():
        raise FileNotFoundError(f"STL-Datei nicht gefunden: {pfad_teil1}")

    # Mesh laden und ein paar Infos ausgeben
    m = load_mesh(str(pfad_teil1))
    # Scale STL numbers from inch to mm.
    m.apply_scale(1.0 / 25.4)
    # Flip Z to match MeshLab orientation (z up)
    flip_z = np.eye(4)
    flip_z[2, 2] = -1.0
    m.apply_transform(flip_z)
    print("Mesh geladen!")

    nozzle_positions = []
    for d in düsen.values():
        p = np.array(d.pos, dtype=float)
        nozzle_positions.append(p)
    nozzle_markers = create_nozzle_cylinders(nozzle_positions)

    rotations = positionieren24()   # Liste mit 24 Matrizen (3x3)

    out_dir = Path(r"C:\Users\micha\Desktop\Bachelorarbeit\Programmierung\Cad Modelle\Modelle1")
    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = pfad_teil1.stem

    positionen_koordinaten = []
    for position_id, (R3, label) in enumerate(rotations, start=1):
        T = np.eye(4)        # 4x4 Einheitsmatrix
        T[:3, :3] = R3       # oben links die 3x3 Rotation einsetzen

        m_rot = m.copy()     # Kopie, Original bleibt unverändert
        m_rot.apply_transform(T)

        # Ecke "rechts-hinten-unten" (E5) der aktuellen Position auf den Ursprung verschieben
        bounds = m_rot.bounds
        minx, miny, minz = bounds[0]
        maxx, maxy, maxz = bounds[1]
        e5 = np.array([maxx, miny, minz], dtype=float)
        T_shift = np.eye(4)
        T_shift[:3, 3] = -e5
        m_rot.apply_transform(T_shift)

        coords = ecken_und_kanten_aus_bounds(m_rot.bounds)
        positionen_koordinaten.append(
            {"position_id": position_id, "label": label, **coords}
        )

        if nozzle_markers is not None:
            m_export = trimesh.util.concatenate([m_rot, nozzle_markers])
        else:
            m_export = m_rot

        out_path = out_dir / f"{base_name}_pos{position_id:02d}_{label}.stl"
        m_export.export(out_path)
        print("Exportiert:", out_path)

    if positionen_koordinaten:
        pos1 = positionen_koordinaten[0]
        print(f"Pos{pos1['position_id']:02d} ({pos1['label']}) Ecken:")
        for idx, e in enumerate(pos1["ecken"], start=1):
            print(f"  E{idx}: {e}")
        print("Duesen (Weltkoordinaten):")
        for name in sorted(düsen.keys()):
            d = düsen[name]
            print(f"  {d.name}: {d.pos} ({d.kraft})")

    # Anzeige entfernt; Export der Dateien reicht.
