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
    "DüseU1": Düse("DüseU1", (-10.0, 0.0, 10.0), "u"),
    "DüseU2": Düse("DüseU2", (-20.0, 0.0, 10.0), "u"),
    "DüseU3": Düse("DüseU3", (-30.0, 0.0, 10.0), "u"),
    "DüseU4": Düse("DüseU4", (-10.0, 0.0, 20.0), "u"),
    "DüseU5": Düse("DüseU5", (-20.0, 0.0, 20.0), "u"),
    "DüseU6": Düse("DüseU6", (-30.0, 0.0, 20.0), "u"),
    "DüseU7": Düse("DüseU7", (-10.0, 0.0, 30.0), "u"),
    "DüseU8": Düse("DüseU8", (-20.0, 0.0, 30.0), "u"),

    "DüseO1": Düse("DüseO1", (-10.0, 10.0, 0.0), "s"),
    "DüseO2": Düse("DüseO2", (-20.0, 10.0, 0.0), "s"),
    "DüseO3": Düse("DüseO3", (-30.0, 10.0, 0.0), "s"),
    "DüseO4": Düse("DüseO4", (-10.0, 20.0, 0.0), "s"),
    "DüseO5": Düse("DüseO5", (-20.0, 20.0, 0.0), "s"),
    "DüseO6": Düse("DüseO6", (-30.0, 20.0, 0.0), "s"),
    "DüseO7": Düse("DüseO7", (-10.0, 30.0, 0.0), "s"),
    "DüseO8": Düse("DüseO8", (-20.0, 30.0, 0.0), "s"),
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


# reale Kanten und Ecken werden bestimmt und Datenstruktur wird angelegt
def echte_kanten_und_ecken(
    mesh: trimesh.Trimesh, angle_deg: float = 90.0, tol_deg: float = 1.0
) -> dict:
    fa_edges = mesh.face_adjacency_edges
    fa_angles = mesh.face_adjacency_angles
    if fa_edges is None or len(fa_edges) == 0:
        return {"edge_indices": np.empty((0, 2), dtype=int), "corner_indices": np.array([], dtype=int),
                "edge_coords": [], "corner_coords": []}
    diff = np.abs(np.degrees(fa_angles) - angle_deg)
    sharp = fa_edges[diff <= tol_deg]
    boundary = mesh.edges_boundary if hasattr(mesh, "edges_boundary") else np.empty((0, 2), dtype=int)
    if boundary is not None and len(boundary) > 0:
        sharp = np.vstack([sharp, boundary])
    if len(sharp) == 0:
        return {"edge_indices": np.empty((0, 2), dtype=int), "corner_indices": np.array([], dtype=int),
                "edge_coords": [], "corner_coords": []}
    sharp = np.unique(np.sort(sharp, axis=1), axis=0)
    corner_idx = np.unique(sharp)
    edge_coords = [(mesh.vertices[a], mesh.vertices[b]) for a, b in sharp]
    corner_coords = [mesh.vertices[i] for i in corner_idx]
    return {
        "edge_indices": sharp,
        "corner_indices": corner_idx,
        "edge_coords": edge_coords,
        "corner_coords": corner_coords,
    }

def create_nozzle_cylinders(positions: list[np.ndarray]) -> trimesh.Trimesh | None:
    if not positions:
        return None
    spheres = []
    for pos in positions:
        sphere = trimesh.creation.icosphere(
            subdivisions=2,
            radius=CylinderRadius,
        )
        sphere.apply_translation(pos)
        spheres.append(sphere)
    return trimesh.util.concatenate(spheres)

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

    echte_geo = echte_kanten_und_ecken(m, angle_deg=90.0, tol_deg=1.0)  #print für die Konsole
    print(f"Echte Kanten: {len(echte_geo['edge_indices'])}")
    print(f"Echte Ecken: {len(echte_geo['corner_indices'])}")
    print("Echte Ecken (Koordinaten):")
    for idx, p in enumerate(echte_geo["corner_coords"], start=1):
        p_rounded = tuple(round(v, 5) for v in p)
        print(f"  K{idx}: {p_rounded}")


    print("Duesen (Weltkoordinaten):") # print Koordinaten der Düsen
    for name in sorted(düsen.keys()):
        d = düsen[name]
        print(f"  {d.name}: {d.pos} ({d.kraft})")

    nozzle_positions = []                 #Düsen-Koordinaten  aus dict werden zu Zylindern
    for d in düsen.values():
        p = np.array(d.pos, dtype=float)
        nozzle_positions.append(p)
    nozzle_markers = create_nozzle_cylinders(nozzle_positions)

    rotations = positionieren24()   # Liste mit 24 Matrizen (3x3)

    out_dir = Path(r"C:\Users\micha\Desktop\Bachelorarbeit\Programmierung\Cad Modelle\Modelle1")
    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = pfad_teil1.stem

    positionen_koordinaten = []      #Liste für alle 24 Positionen wird erstellt 
    for position_id, (R3, label) in enumerate(rotations, start=1):
        T = np.eye(4)        # 4x4 Einheitsmatrix
        T[:3, :3] = R3       # oben links die 3x3 Rotation einsetzen   Mesh wird rotiert

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




        echte = echte_kanten_und_ecken(m_rot, angle_deg=90.0, tol_deg=1.0) # Berechnet echten 90° Kanten Ecken für die aktuelle Positione und speichert sie
        positionen_koordinaten.append(
            {
                "position_id": position_id,
                "label": label,
                "ecken": echte["corner_coords"],
                "kanten": echte["edge_indices"],
                "kanten_coords": echte["edge_coords"],
            }
        )
         



        if nozzle_markers is not None:  #Exportiert alle Positionen
            m_export = trimesh.util.concatenate([m_rot, nozzle_markers])
        else:
            m_export = m_rot

        out_path = out_dir / f"{base_name}_pos{position_id:02d}_{label}.stl"
        m_export.export(out_path)
        print("Exportiert:", out_path)


        
    # Anzeige entfernt; Export der Dateien reicht.

e1 = positionen_koordinaten[0]["ecken"][0][0] 
print(e1)
#positionen_koordinaten[0] = Pos1
#["ecken"][0] = Ecke E1
#["ecken"][0][0] = Ecke E1 x-Wert

