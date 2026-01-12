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

def positionieren24() -> list[tuple[np.ndarray, str]]: #Erstellt die Positioen
    """
    Liefert 24 Rotationen in einer einfachen, logischen Reihenfolge:
    - Position 1..4: nur z-Drehungen (0/90/180/270)
    - Position 5..8: nÃ¤chste "Grundlage" + wieder 0/90/180/270 um z
    - usw.
    """
    I = np.eye(3)  # Einheitsmatrix wird erstellt

    # 6 Grundlagen = welche Seite "unten" ist (vereinfachtes, gut verstÃ¤ndliches Set)
    bases = [    # alle 6 Flächen liegen einmal unten Ausgangsfläche, die im nächsten Schritt dann um die z-Achse gedreht werden. So erhalte ich alle Positionen
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

#Bestimmt welche Düse zum Kippen zu gebrauchen ist
def kipp_duse(kipp_in: str, schwerpunkt: Vec3) -> list[tuple[Duse, float, float | None, float | None]] | None:    #Kippmöglichkeiten   schwerpunkt wird als transofmirten Vector ausgegeben x, y, z
    if kipp_in == "o_x_p":
        funk_duesen = []                    # Liste mit Düsen die zum Kippen verwendet werden können d: das Duse-Objekt  dx: Abstand in x zum Schwerpunkt   dz: Abstand in z um Schwerpunkt
        for d in duesen.values():
            if d.kraft == "o" and d.pos[1] > schwerpunkt[1]: 
                dx = abs(d.pos[0] - schwerpunkt[0])     #Abstand zwischen Düse und Schwerpunkt        abs() gibt den Betrag aus
                dy = abs(d.pos[1] - schwerpunkt[1])     # 0 = x, 1 = y, 2 = z
                dz = None
                funk_duesen.append((d, dx,dy, dz))
        neue_liste = []
        for d, dx,dy, dz in funk_duesen:  
            if dx < dy:
                neue_liste.append((d, dx,dy, dz))  
        funk_duesen = neue_liste  # Überschreibt alte Liste
        return funk_duesen
    
    if kipp_in == "o_x_n":                              
        funk_duesen = []                    # Liste mit Düsen die zum Kippen verwendet werden können d: das Duse-Objekt  dx: Abstand in x zum Schwerpunkt   dz: Abstand in z um Schwerpunkt
        for d in duesen.values():
            if d.kraft == "o" and d.pos[1] < schwerpunkt[1]:
                dx = abs(d.pos[0] - schwerpunkt[0])     #Abstand zwischen Düse und Schwerpunkt        abs() gibt den Betrag aus
                dy = abs(d.pos[1] - schwerpunkt[1])     # 0 = x, 1 = y, 2 = z
                dz = None
                funk_duesen.append((d, dx,dy, dz))
        neue_liste = []
        for d, dx,dy, dz in funk_duesen:  
            if dx < dy:
                neue_liste.append((d, dx,dy, dz))  
        funk_duesen = neue_liste  # Überschreibt alte Liste
        return funk_duesen
    
    if kipp_in == "o_y_p":
        funk_duesen = []                    # Liste mit Düsen die zum Kippen verwendet werden können d: das Duse-Objekt  dx: Abstand in x zum Schwerpunkt   dz: Abstand in z um Schwerpunkt
        for d in duesen.values():
            if d.kraft == "o" and d.pos[0] < schwerpunkt[0]: 
                dx = abs(d.pos[0] - schwerpunkt[0])     #Abstand zwischen Düse und Schwerpunkt        abs() gibt den Betrag aus
                dy = abs(d.pos[1] - schwerpunkt[1])     # 0 = x, 1 = y, 2 = z
                dz = None
                funk_duesen.append((d, dx,dy, dz))
        neue_liste = []
        for d, dx,dy, dz in funk_duesen:   
            if dx > dy:
                neue_liste.append((d, dx,dy, dz))  
        funk_duesen = neue_liste  # Überschreibt alte Liste
        return funk_duesen
    if kipp_in == "o_y_n":
        funk_duesen = []                    # Liste mit Düsen die zum Kippen verwendet werden können d: das Duse-Objekt  dx: Abstand in x zum Schwerpunkt   dz: Abstand in z um Schwerpunkt
        for d in duesen.values():
            if d.kraft == "o" and d.pos[0] > schwerpunkt[0]: 
                dx = abs(d.pos[0] - schwerpunkt[0])     #Abstand zwischen Düse und Schwerpunkt        abs() gibt den Betrag aus
                dy = abs(d.pos[1] - schwerpunkt[1])     # 0 = x, 1 = y, 2 = z
                dz = None
                funk_duesen.append((d, dx,dy, dz))
        neue_liste = []
        for d, dx,dy, dz in funk_duesen:   
            if dx > dy:
                neue_liste.append((d, dx,dy, dz))  
        funk_duesen = neue_liste  # Überschreibt alte Liste
        return funk_duesen
    if kipp_in == "u_x_p":
        funk_duesen = []                    # Liste mit Düsen die zum Kippen verwendet werden können d: das Duse-Objekt  dx: Abstand in x zum Schwerpunkt   dz: Abstand in z um Schwerpunkt
        for d in duesen.values():
            if d.kraft == "u" and d.pos[2] < schwerpunkt[2]: 
                dx = abs(d.pos[0] - schwerpunkt[0])     #Abstand zwischen Düse und Schwerpunkt        abs() gibt den Betrag aus
                dz = abs(d.pos[2] - schwerpunkt[2])   # 0 = x, 1 = y, 2 = z
                dy= None
                funk_duesen.append((d, dx,dy, dz))
        neue_liste = []
        for d, dx,dy, dz in funk_duesen:  
            if dx < dz:
                neue_liste.append((d, dx,dy, dz))  
        funk_duesen = neue_liste  # Überschreibt alte Liste
        return funk_duesen
    
    if kipp_in == "u_x_n":
        funk_duesen = []                    # Liste mit Düsen die zum Kippen verwendet werden können d: das Duse-Objekt  dx: Abstand in x zum Schwerpunkt   dz: Abstand in z um Schwerpunkt
        for d in duesen.values():
            if d.kraft == "u" and d.pos[2] > schwerpunkt[2]: # Nur Düsen mit u und der z Wert der Düsen muss größer sein als der z Wert des Schwerpunkts
                dx = abs(d.pos[0] - schwerpunkt[0])     #Abstand zwischen Düse und Schwerpunkt        abs() gibt den Betrag aus
                dz = abs(d.pos[2] - schwerpunkt[2])   # 0 = x, 1 = y, 2 = z
                dy = None
                funk_duesen.append((d, dx,dy, dz))
        neue_liste = []
        for d, dx,dy, dz in funk_duesen:  # Fügt nur Objekte zur neuen Liste hinzu wenn x<z 
            if dx < dz:
                neue_liste.append((d, dx,dy, dz))  
        funk_duesen = neue_liste  # Überschreibt alte Liste
        return funk_duesen
    
    if kipp_in == "u_z_p":
        funk_duesen = []                    # Liste mit Düsen die zum Kippen verwendet werden können d: das Duse-Objekt  dx: Abstand in x zum Schwerpunkt   dz: Abstand in z um Schwerpunkt
        for d in duesen.values():
            if d.kraft == "u" and d.pos[0] > schwerpunkt[0]: 
                dx = abs(d.pos[0] - schwerpunkt[0])     #Abstand zwischen Düse und Schwerpunkt        abs() gibt den Betrag aus
                dz = abs(d.pos[2] - schwerpunkt[2])   # 0 = x, 1 = y, 2 = z
                dy = None
                funk_duesen.append((d, dx,dy, dz))
        neue_liste = []
        for d, dx, dy, dz in funk_duesen:  
            if dx > dz:
                neue_liste.append((d, dx,dy, dz))  
        funk_duesen = neue_liste  # Überschreibt alte Liste
        return funk_duesen
    
    if kipp_in == "u_z_n":
        funk_duesen = []                    # Liste mit Düsen die zum Kippen verwendet werden können d: das Duse-Objekt  dx: Abstand in x zum Schwerpunkt   dz: Abstand in z um Schwerpunkt
        for d in duesen.values():
            if d.kraft == "u" and d.pos[0] < schwerpunkt[0]: 
                dx = abs(d.pos[0] - schwerpunkt[0])     #Abstand zwischen Düse und Schwerpunkt        abs() gibt den Betrag aus
                dz = abs(d.pos[2] - schwerpunkt[2])   # 0 = x, 1 = y, 2 = z
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

def create_nozzle_cylinders(positions: list[np.ndarray]) -> trimesh.Trimesh | None:  #Zylinder, die Düsen symbolisieren für die stl-Datei
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

    echte_geo = echte_kanten_und_ecken(m, angle_deg=90.0, tol_deg=1.0)  #print für die Konsole der Ausgangskoordinaten nicht transformiert
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

    positionen_koordinaten = []      #Liste für alle 24 Positionen wird erstellt 

    Schwerpunkt = tuple(round(v, 5) for v in m.center_mass) # Ausgangsschwerpunkt wird berechnet 

    for position_id, (R3, label) in enumerate(rotations, start=1):  #Transformation beginnt erst wird das Mesh pro Position orientiert dann verschoben
        T = np.eye(4)        # 4x4 Einheitsmatrix
        T[:3, :3] = R3       # oben links die 3x3 Rotation einsetzen   Mesh wird orientiert

        m_ver_orie = m.copy()     # Kopie, Original bleibt unverändert
        m_ver_orie.apply_transform(T)  #mesh wird durch T orientiert

        # Ecke "rechts-hinten-unten" (E5) der aktuellen Position auf den Ursprung verschieben
        #Verschiebt Bounding-Box 
        bounds = m_ver_orie.bounds
        minx, miny, minz = bounds[0]
        maxx, maxy, maxz = bounds[1]
        e5 = np.array([maxx, miny, minz], dtype=float)
        T_shift = np.eye(4)
        T_shift[:3, 3] = -e5
        m_ver_orie.apply_transform(T_shift) #mesh wird durch T_shift verschoben
        #m_ver_orie ist das verschobenen und orientierte Mesh

        schwerpunkt_h = np.array([*Schwerpunkt, 1.0], dtype=float)  
        schwerpunkt_rot = (T_shift @ T @ schwerpunkt_h)[:3] #Der Schwerpunkt wird rotiert und verschoben

        # Speichert Transformation und Mesh-Daten für spätere Ray-Test/Strahl (Startpunkt unterhalb des Meshes).
        T_total = T_shift @ T
        bounds_sel = m_ver_orie.bounds
        y_min = bounds_sel[0][1] - 1.0
        z_min = bounds_sel[0][2] - 1.0
        triangles_sel = m_ver_orie.triangles

        echte =  echte_kanten_und_ecken(m_ver_orie, angle_deg=90.0, tol_deg=1.0) # transormierte echte Ecken und Kanten aus der Geometrie
        #echte_kanten_und_ecken gibt die Koordinaten der Ecken und Kanten soei die Ecken Indizese und die Kanten Indizes
        #echte enthält dann die verschobenen und orientierten Koordianten der Ecken und Kanten da in m_ver_orie die verschiebung und orientierung enthalten ist
        
        positionen_koordinaten.append(  #Für jede Position wird ein Dictionary angelget mit den relevanten Daten
            {  #benötige ich, wenn ich auf z.B Eckkordinaten einer bestimmten Position zugreifen will
                "position_id": position_id,
                "label": label,
                "schwerpunkt": tuple(round(v, 5) for v in schwerpunkt_rot),
                "ecken": echte["corner_coords"],
                "kanten": echte["edge_indices"],
                "kanten_coords": echte["edge_coords"],
                
                # Daten für Ray-Tests auf der Oberfläche der aktuellen Position
                "T_total": T_total,
                "bounds": bounds_sel,
                "y_min": y_min,
                "z_min": z_min,
                "triangles": triangles_sel,
            }
        )
         

        #Legt Ordner fest für die exportierten stl-Dateien
        out_dir = Path(r"C:\Users\micha\Desktop\Bachelorarbeit\Programmierung\Cad Modelle\Modelle1") 
        out_dir.mkdir(parents=True, exist_ok=True)
        base_name = pfad_teil1.stem
        if nozzle_markers is not None:  #Exportiert alle Positionen in stl-Dateien
            m_export = trimesh.util.concatenate([m_ver_orie, nozzle_markers])
        else:
            m_export = m_ver_orie

        out_path = out_dir / f"{base_name}_pos{position_id:02d}_{label}.stl"
        m_export.export(out_path)
        print("Exportiert:", out_path)



     # Position abfragen und Koordinatne der Ecken ausgeben 
    user_in = input("Welche Position fuer Ecken? (1-24): ").strip()
    pos_id = int(user_in)
    pos = positionen_koordinaten[pos_id - 1] #Holt sich die Information aus dem Dictor über die Positionen. position_id label schwerpunkt ecken, kanten, kanten_coords
    schwerpunkt_pos = pos["schwerpunkt"]  #Transformierte Schwerpunktkoordinaten werden aus dem Dict geholt


    T_total = pos["T_total"]
    bounds_sel = pos["bounds"]
    y_min = pos["y_min"]
    z_min = pos["z_min"]
    triangles_sel = pos["triangles"]
    m_sel = m.copy()
    m_sel.apply_transform(T_total)


    print("Pos", pos["position_id"], "Ecken:") #Terminal Text
    for idx, e in enumerate(pos["ecken"]): #enumerate gibt idx und e aus
        e_rounded = (
            round(e[0], 5),
            round(e[1], 5),
            round(e[2], 5),
        )
        print(f"  {idx} {e_rounded}")  # Gibt den Index aus also z.b 0 und e_rounded also die gerundeten Eckenwerte aus

    print("Welche kipp_duse? (o_x_p, o_x_n, o_y_p, o_y_n, u_x_p, u_x_n, u_z_p, u_z_n): ")
    kipp_in = input().strip()
    funkduse = kipp_duse(kipp_in, schwerpunkt_pos) #kipp_duse wird aufgerufen der schwerpunkt wird übergeben und funk_dusen wird zurückgegeben
    #Bestimmt welche Düse zum Kippen zu gebrauchen ist
    print("Schwerpunkt transformiert:", schwerpunkt_pos)
   


    print(f"kipp_duse {kipp_in} (unter Schwerpunkt):")
    for d, dx, dy, dz in funkduse:
        print(f"  {d.name}: {d.pos} | dx={dx} dy={dy} dz={dz}")


        if d.kraft == "u":
            origin = np.array([d.pos[0], y_min, d.pos[2]], dtype=float)  #Setzt den Startpunkt des Strahls gleich x z der Düse
            direction = np.array([0.0, 1.0, 0.0], dtype=float) #Richtung, in die der Strahl zeigt
            locations, _, _ = m_sel.ray.intersects_location([origin], [direction])
            print("    xz_surface:", len(locations) > 0)
        elif d.kraft == "o":
            origin = np.array([d.pos[0], d.pos[1], z_min], dtype=float)
            direction = np.array([0.0, 0.0, 1.0], dtype=float)
            locations, _, _ = m_sel.ray.intersects_location([origin], [direction])
            print("    xy_surface:", len(locations) > 0)
    
       

