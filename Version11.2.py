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
class Duse:
    name: str
    pos: Vec3
    kraft: str  # "u" = unten, "s" = seite


# Koordinatensystem-Konvention:
# x = vorne, y = links, z = oben
Ursprung: Vec3 = (0.0, 0.0, 0.0)

# Dusen "Datenbank"
# Zugriff z.B.: print(duesen["DuseU1"].pos[0])
duesen: Dict[str, Duse] = {   #   x    y    z
    "DuseU1": Duse("DuseU1", (-10.0, 0.0, 10.0), "u"),
    "DuseU2": Duse("DuseU2", (-20.0, 0.0, 10.0), "u"),
    "DuseU3": Duse("DuseU3", (-30.0, 0.0, 10.0), "u"),
    "DuseU4": Duse("DuseU4", (-10.0, 0.0, 20.0), "u"),
    "DuseU5": Duse("DuseU5", (-20.0, 0.0, 20.0), "u"),
    "DuseU6": Duse("DuseU6", (-30.0, 0.0, 20.0), "u"),
    "DuseU7": Duse("DuseU7", (-10.0, 0.0, 30.0), "u"),
    "DuseU8": Duse("DuseU8", (-20.0, 0.0, 30.0), "u"),

    "DuseO1": Duse("DuseO1", (-10.0, 10.0, 0.0), "o"),
    "DuseO2": Duse("DuseO2", (-20.0, 10.0, 0.0), "o"),
    "DuseO3": Duse("DuseO3", (-30.0, 10.0, 0.0), "o"),
    "DuseO4": Duse("DuseO4", (-10.0, 20.0, 0.0), "o"),
    "DuseO5": Duse("DuseO5", (-20.0, 20.0, 0.0), "o"),
    "DuseO6": Duse("DuseO6", (-30.0, 20.0, 0.0), "o"),
    "DuseO7": Duse("DuseO7", (-10.0, 30.0, 0.0), "o"),
    "DuseO8": Duse("DuseO8", (-20.0, 30.0, 0.0), "o"),
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


##neu

def point_in_triangle_2d(     #Funktion, die Prüft ob ein PUnkt in einem 2D-Dreieck liegt.
    p: tuple[float, float],   #Zu prüfender Punkt
    tri: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],  #Eckpunkte des Dreiecks
    eps: float = 1e-6, #Rundungsfehler
) -> bool:                    #Gibt true oder false zurück
    def sign(p1, p2, p3): #p1 zu prüfender Punkt
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])  #Sagt auf welcher Seite der Punkt p1 relativ zu den Katen liegt
                                                                                      # Ergebnis > 0 Punkt liegt auf einer Seite
                                                                                      # Ergebnis < 0 Punkt liegt auf der anderen Seite
                                                                                      # Ergebnis = 0 Punkt liegt auf beiden Seiten
    d1 = sign(p, tri[0], tri[1])                      
    d2 = sign(p, tri[1], tri[2])
    d3 = sign(p, tri[2], tri[0])        #Befindet sich der Punkt immer auf der gleichen Seite der Kanten. Wenn ja, ist der Punkt im Dreieck

    has_neg = (d1 < -eps) or (d2 < -eps) or (d3 < -eps)  # Prüft, ob Vorzeichen gemischt sind    Wird negativ, wenn mindestens ein Wert negativ ist
    has_pos = (d1 > eps) or (d2 > eps) or (d3 > eps) #Wird positiv, wenn mindestens ein Wert positiv ist
    return not (has_neg and has_pos) # Entscheidung, wenn positive und negative Werte vorkommen dann liegt der Punkt außerhalb, weil er auf verschiedenen Seiten der Dreieckskanten liegt
                                     # Wenn nur negative oder nur positive Werte vorkommen, dann liegt der PUnkt innerhalb 
##neu


def kipp_duse(kipp_in: str, schwerpunkt: Vec3, transform: np.ndarray | None = None) -> str | None:    #Kippmöglichkeiten    ####Macht str hier sinn? besser float?
    duesen_pos = []
    for d in duesen.values():   #Positionsabhängigkeit des Schwerpunkts 
        p_h = np.array([d.pos[0], d.pos[1], d.pos[2], 1.0], dtype=float)
        if transform is not None:
            p_h = transform @ p_h
        p = p_h[:3]
        duesen_pos.append((d, p))


    if kipp_in == "o_x_p":
        funk_duesen = []                    # Liste mit Düsen die zum Kippen verwendet werden können d: das Duse-Objekt  dx: Abstand in x zum schwerpunkt   dz: Abstand in z um schwerpunkt
        for d, p in duesen_pos:
            if d.kraft == "o" and p[1] > schwerpunkt[1]: 
                dx = abs(p[0] - schwerpunkt[0])     #Abstand zwischen Düse und schwerpunkt        abs() gibt den Betrag aus
                dy = abs(p[1] - schwerpunkt[1])     # 0 = x, 1 = y, 2 = z
                dz = None
                funk_duesen.append((d, dx,dy, dz))
        neue_liste = []
        for d, dx,dy, dz in funk_duesen:  
            if dx < dy:
                neue_liste.append((d, dx,dy, dz))  
        funk_duesen = neue_liste  # Überschreibt alte Liste
        return funk_duesen
    
    if kipp_in == "o_x_n":                              
        funk_duesen = []                    # Liste mit Düsen die zum Kippen verwendet werden können d: das Duse-Objekt  dx: Abstand in x zum schwerpunkt   dz: Abstand in z um schwerpunkt
        for d, p in duesen_pos:
            if d.kraft == "o" and p[1] < schwerpunkt[1]:
                dx = abs(p[0] - schwerpunkt[0])     #Abstand zwischen Düse und schwerpunkt        abs() gibt den Betrag aus
                dy = abs(p[1] - schwerpunkt[1])     # 0 = x, 1 = y, 2 = z
                dz = None
                funk_duesen.append((d, dx,dy, dz))
        neue_liste = []
        for d, dx,dy, dz in funk_duesen:  
            if dx < dy:
                neue_liste.append((d, dx,dy, dz))  
        funk_duesen = neue_liste  # Überschreibt alte Liste
        return funk_duesen
    
    if kipp_in == "o_y_p":
        funk_duesen = []                    # Liste mit Düsen die zum Kippen verwendet werden können d: das Duse-Objekt  dx: Abstand in x zum schwerpunkt   dz: Abstand in z um schwerpunkt
        for d, p in duesen_pos:
            if d.kraft == "o" and p[0] < schwerpunkt[0]: 
                dx = abs(p[0] - schwerpunkt[0])     #Abstand zwischen Düse und schwerpunkt        abs() gibt den Betrag aus
                dy = abs(p[1] - schwerpunkt[1])     # 0 = x, 1 = y, 2 = z
                dz = None
                funk_duesen.append((d, dx,dy, dz))
        neue_liste = []
        for d, dx,dy, dz in funk_duesen:   
            if dx > dy:
                neue_liste.append((d, dx,dy, dz))  
        funk_duesen = neue_liste  # Überschreibt alte Liste
        return funk_duesen
    if kipp_in == "o_y_n":
        funk_duesen = []                    # Liste mit Düsen die zum Kippen verwendet werden können d: das Duse-Objekt  dx: Abstand in x zum schwerpunkt   dz: Abstand in z um schwerpunkt
        for d, p in duesen_pos:
            if d.kraft == "o" and p[0] > schwerpunkt[0]: 
                dx = abs(p[0] - schwerpunkt[0])     #Abstand zwischen Düse und schwerpunkt        abs() gibt den Betrag aus
                dy = abs(p[1] - schwerpunkt[1])     # 0 = x, 1 = y, 2 = z
                dz = None
                funk_duesen.append((d, dx,dy, dz))
                print(d.name, "p=", p, "dx=", dx, "dy=", dy, "x>cm?", p[0] > schwerpunkt[0], "dx>dy?", dx > dy)
                

        neue_liste = []
        for d, dx,dy, dz in funk_duesen:   
            if dx < dy:                                ##Hier könnte der Fehler liegen
                neue_liste.append((d, dx,dy, dz))  
                
        funk_duesen = neue_liste  # Überschreibt alte Liste
        
        return funk_duesen
    if kipp_in == "u_x_p":
        funk_duesen = []                    # Liste mit Düsen die zum Kippen verwendet werden können d: das Duse-Objekt  dx: Abstand in x zum schwerpunkt   dz: Abstand in z um schwerpunkt
        for d, p in duesen_pos:
            if d.kraft == "u" and p[2] < schwerpunkt[2]: 
                dx = abs(p[0] - schwerpunkt[0])     #Abstand zwischen Düse und schwerpunkt        abs() gibt den Betrag aus
                dz = abs(p[2] - schwerpunkt[2])   # 0 = x, 1 = y, 2 = z
                dy= None
                funk_duesen.append((d, dx,dy, dz))
        neue_liste = []
        for d, dx,dy, dz in funk_duesen:  
            if dx < dz:
                neue_liste.append((d, dx,dy, dz))  
        funk_duesen = neue_liste  # Überschreibt alte Liste
        return funk_duesen
    
    if kipp_in == "u_x_n":
        funk_duesen = []                    # Liste mit Düsen die zum Kippen verwendet werden können d: das Duse-Objekt  dx: Abstand in x zum schwerpunkt   dz: Abstand in z um schwerpunkt
        for d, p in duesen_pos:
            if d.kraft == "u" and p[2] > schwerpunkt[2]: # Nur Düsen mit u und der z Wert der Düsen muss größer sein als der z Wert des schwerpunkts
                dx = abs(p[0] - schwerpunkt[0])     #Abstand zwischen Düse und schwerpunkt        abs() gibt den Betrag aus
                dz = abs(p[2] - schwerpunkt[2])   # 0 = x, 1 = y, 2 = z
                dy = None
                funk_duesen.append((d, dx,dy, dz))
        neue_liste = []
        for d, dx,dy, dz in funk_duesen:  # Fügt nur Objekte zur neuen Liste hinzu wenn x<z 
            if dx < dz:
                neue_liste.append((d, dx,dy, dz))  
        funk_duesen = neue_liste  # Überschreibt alte Liste
        return funk_duesen
    
    if kipp_in == "u_z_p":
        funk_duesen = []                    # Liste mit Düsen die zum Kippen verwendet werden können d: das Duse-Objekt  dx: Abstand in x zum schwerpunkt   dz: Abstand in z um schwerpunkt
        for d, p in duesen_pos:
            if d.kraft == "u" and p[0] > schwerpunkt[0]: 
                dx = abs(p[0] - schwerpunkt[0])     #Abstand zwischen Düse und schwerpunkt        abs() gibt den Betrag aus
                dz = abs(p[2] - schwerpunkt[2])   # 0 = x, 1 = y, 2 = z
                dy = None
                funk_duesen.append((d, dx,dy, dz))
        neue_liste = []
        for d, dx, dy, dz in funk_duesen:  
            if dx > dz:
                neue_liste.append((d, dx,dy, dz))  
        funk_duesen = neue_liste  # Überschreibt alte Liste
        return funk_duesen
    
    if kipp_in == "u_z_n":
        funk_duesen = []                    # Liste mit Düsen die zum Kippen verwendet werden können d: das Duse-Objekt  dx: Abstand in x zum schwerpunkt   dz: Abstand in z um schwerpunkt
        for d, p in duesen_pos:
            if d.kraft == "u" and p[0] < schwerpunkt[0]: 
                dx = abs(p[0] - schwerpunkt[0])     #Abstand zwischen Düse und schwerpunkt        abs() gibt den Betrag aus
                dz = abs(p[2] - schwerpunkt[2])   # 0 = x, 1 = y, 2 = z
                dy = None
                funk_duesen.append((d, dx,dy, dz))
        neue_liste = []
        for d, dx,dy, dz in funk_duesen:  
            if dx > dz:
                neue_liste.append((d, dx,dy, dz))  
        funk_duesen = neue_liste  # Überschreibt alte Liste
        return funk_duesen
    
    
### Aktuell werden noch Düsen außerhalb des Teils zurückgegeben 
### x und y Werte der Düse müssen genau so auch im Objekt vorkommen  für oben 
## Schwerpunkt Transformation funktioniert nicht

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


    Schwerpunkt = tuple(round(v, 5) for v in m.center_mass) # Schwerpunkt berechnen
    if not m.is_watertight:
        print("Hinweis: Mesh ist nicht wasserdicht, Schwerpunkt kann ungenau sein.")

    echte_geo = echte_kanten_und_ecken(m, angle_deg=90.0, tol_deg=1.0)  #print für die Konsole
    print(f"Echte Kanten: {len(echte_geo['edge_indices'])}")
    print(f"Echte Ecken: {len(echte_geo['corner_indices'])}")
    print("Echte Ecken (Koordinaten):")
    for idx, p in enumerate(echte_geo["corner_coords"], start=1):
        p_rounded = tuple(round(v, 5) for v in p)
        print(f"  K{idx}: {p_rounded}")


    print("Duesen (Weltkoordinaten):") # print Koordinaten der Dusen
    for name in sorted(duesen.keys()):
        d = duesen[name]
        print(f"  {d.name}: {d.pos} ({d.kraft})")

    nozzle_positions = []                 #Dusen-Koordinaten  aus dict werden zu Zylindern
    for d in duesen.values():
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
        T_total = T_shift @ T   ##neu kombiniert die T Rotation und T_shift Verschiebe Matrix




        echte = echte_kanten_und_ecken(m_rot, angle_deg=90.0, tol_deg=1.0) # Berechnet echten 90° Kanten Ecken für die aktuelle Positione und speichert sie
        center_mass_pos = tuple(round(v, 5) for v in m_rot.center_mass)
        vertices_xy = {(round(v[0], 5), round(v[1], 5)) for v in m_rot.vertices} #Speichert alle Vertex-Punkte(Eckpunkte des Meshs. Ein Mesh besteht aus Dreiecken und jedes Dreieck hat 3 Vertex-Punkte)
        triangles_xy = []
        triangles_xz = []
        for t, n in zip(m_rot.triangles, m_rot.face_normals):
            if abs(n[2]) > 0.9:  # fast parallel zur XY-Ebene
                triangles_xy.append(
                    (
                        (float(t[0][0]), float(t[0][1])),   #Speichert nur die xy Werte der Dreiecke
                        (float(t[1][0]), float(t[1][1])),
                        (float(t[2][0]), float(t[2][1])),
                    )
                )
            if abs(n[1]) > 0.9:  # fast parallel zur XZ-Ebene
                triangles_xz.append(
                    (
                        (float(t[0][0]), float(t[0][2])), #Speichert nur die xz Werte der Dreiecke
                        (float(t[1][0]), float(t[1][2])),
                        (float(t[2][0]), float(t[2][2])),
                    )
                )
        positionen_koordinaten.append(
            {
                "position_id": position_id,        #24 Positionen
                "label": label,                    #x0_z0 Name der Position
                "ecken": echte["corner_coords"],   #Speichert die Koordinaten der Eckpunkte
                "kanten": echte["edge_indices"],   #Speichert die Indizies der Kanten (Vertex-Paare)
                "kanten_coords": echte["edge_coords"], #Speicher die Koordinaten der Kanten 
                "vertices_xy": vertices_xy,        #XY Vertex Punkte. Damit kann ich überprüfen, ob die Düsen exakt auf einem vorhandne Mesh Vertex liegt
                "triangles_xy": triangles_xy,      #alle Dreicke als xy Projektion Damit kann ich prüfen, ob ein Düse in einer Fläche des Meshs liegt, nicht auf einem Eckpunkt
                "triangles_xz": triangles_xz,      #xz-Projektion
                "transform": T_total,              #um Düsenpunkte auch in dieselbe Position wie das Mesh transformieren zu können. Kombiniert Rotation und Verschiebung des Mesh. 
                "center_mass": center_mass_pos,    #Schwerpunkt fuer diese Position
                #Um die DüsenKoordinaten verwenden zu können, müssen dies transformiert werden 
            }
        )
         ##neu



        if nozzle_markers is not None:  #Exportiert alle Positionen
            m_export = trimesh.util.concatenate([m_rot, nozzle_markers])
        else:
            m_export = m_rot

        out_path = out_dir / f"{base_name}_pos{position_id:02d}_{label}.stl"
        m_export.export(out_path)
        print("Exportiert:", out_path)


    if positionen_koordinaten:   # Position abfragen und Koordinatne der Ecken ausgeben 
        user_in = input("Welche Position fuer Ecken? (1-24): ").strip()
        try:
            pos_id = int(user_in)
        except ValueError:
            pos_id = 1
        if pos_id < 1 or pos_id > len(positionen_koordinaten):
            pos_id = 1
        pos = positionen_koordinaten[pos_id - 1]
        print(f"Pos{pos['position_id']:02d} ({pos['label']}) Ecken:")
        for idx, e in enumerate(pos["ecken"], start=1):
            e_rounded = tuple(round(v, 5) for v in e)
            print(f"  E{idx}: {e_rounded}")

        kipp_keys = ["o_x_p", "o_x_n", "o_y_p", "o_y_n", "u_x_p", "u_x_n", "u_z_p", "u_z_n"] #Abfrage wie gekippt werden soll
        kipp_in = input(f"Welche kipp_duse? ({', '.join(kipp_keys)}): ").strip()
        funk_duesen = kipp_duse(kipp_in, pos.get("center_mass"), pos.get("transform"))
        print(f"Schwerpunkt (Position): {pos.get('center_mass')}")
        if isinstance(funk_duesen, list): #Prüft, ob funk_duesen eine Liste ist. Brauche ich vielleicht nicht
            print(f"kipp_duse {kipp_in} (vor Trefferpruefung):") #Gibt die Düsen aus, die nach der ersten Auswahl noch in frage kommen. Vor der Mesh auswahl.
            for d, dx, dy, dz in funk_duesen:
                print(f"  {d.name}: {d.pos} | dx={dx} dy={dy} dz={dz}")
            triangles_xy = pos.get("triangles_xy", [])  # neu   Holt die XY-Dreiecke der ausgewaehlten Position
            transform = pos.get("transform")            # Stellt die Transformermatirx bereit
            trefferdusen_o = []                         #Liste mit Düsen die bezug zum Mesh haben 
            trefferdusen_u = []
            if triangles_xy:                            # Sind die Dreiecke vorhanden?
                has_o = any(d.kraft == "o" for d, dx, dy, dz in funk_duesen) #Prueft nach o-Duesen in der Liste
                if has_o:
                    print("Treffer (x,y in Mesh-Flaeche):")
                    for d, dx, dy, dz in funk_duesen:
                        if d.kraft != "o":
                            continue
                        p_h = np.array([d.pos[0], d.pos[1], d.pos[2], 1.0], dtype=float) #Baut die Duese als Homogenen Vektor (x,y,z,1) 4x4  Transformationsmatrxi
                        if transform is not None:  #3x3 Roation und Verschiebung    Transformiert die Duese in das Koordinatensystem der gewaehlten Position. Wenn nicht wuerde die Duese an einer anderen Stelle stehen.
                            p_h = transform @ p_h
                        p_xy = (float(p_h[0]), float(p_h[1]))  #nimmt nur die x und y Koordinate zur Pruefung 
                        hit_xy = any(point_in_triangle_2d(p_xy, tri) for tri in triangles_xy) #Prueft, ob der PUnkt in irgendeinem XY-Dreieck liegt
                        if hit_xy:
                            trefferdusen_o.append((d, dx, dy, dz))
                        else:
                            print(f"DBG kein XY-Treffer: {d.name} p_xy={p_xy}") #debug
            else:
                print("Keine Mesh-Dreiecks-Daten fuer diese Position.")
            triangles_xz = pos.get("triangles_xz", [])
            if triangles_xz:
                has_u = any(d.kraft == "u" for d, dx, dy, dz in funk_duesen)
                if has_u:
                    print("Treffer (x,z in Mesh-Flaeche):")
                    for d, dx, dy, dz in funk_duesen:
                        if d.kraft != "u":
                            continue
                        p_h = np.array([d.pos[0], d.pos[1], d.pos[2], 1.0], dtype=float)
                        if transform is not None:
                            p_h = transform @ p_h
                        p_xz = (float(p_h[0]), float(p_h[2]))
                        if any(point_in_triangle_2d(p_xz, tri) for tri in triangles_xz):
                            trefferdusen_u.append((d, dx, dy, dz))
                            #print(f"  {d.name}: x,z={p_xz}")
            else:
                print("Keine Mesh-Dreiecks-Daten fuer XZ.")
            funk_duesen = trefferdusen_o + trefferdusen_u  #Liste mit Düsen, die im Mesh liegen wird erstellt
            print(f"kipp_duse {kipp_in} (Treffer in Mesh-Flaeche):")
            for d, dx, dy, dz in funk_duesen:
                print(f"  {d.name}: {d.pos} | dx={dx} dy={dy} dz={dz}")
        elif funk_duesen is not None:
            print(f"kipp_duse {kipp_in}: {funk_duesen}")
        else:
            print("Unbekannte kipp_duse-Auswahl.")


    # Anzeige entfernt; Export der Dateien reicht.



