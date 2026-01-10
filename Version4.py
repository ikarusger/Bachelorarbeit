from dataclasses import dataclass
from typing import Dict, Tuple
from pathlib import Path

import numpy as np
import trimesh
from itertools import permutations, product


Vec3 = Tuple[float, float, float]  # (x, y, z)


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

    out_dir = Path(r"C:\Users\micha\Desktop\Bachelorarbeit\Programmierung\Cad Modelle\Modelle1")
    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = pfad_teil1.stem

    for position_id, (R3, label) in enumerate(rotations, start=1):
        T = np.eye(4)        # 4x4 Einheitsmatrix
        T[:3, :3] = R3       # oben links die 3x3 Rotation einsetzen

        m_rot = m.copy()     # Kopie, Original bleibt unverändert
        m_rot.apply_transform(T)

        out_path = out_dir / f"{base_name}_pos{position_id:02d}_{label}.stl"
        m_rot.export(out_path)
        print("Exportiert:", out_path)

    # Anzeige entfernt; Export der Dateien reicht.

    
