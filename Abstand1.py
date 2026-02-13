from dataclasses import dataclass
from typing import Dict, Tuple
from pathlib import Path
import csv

import numpy as np
import trimesh
import math


Vec3 = Tuple[float, float, float]  # (x, y, z)

CylinderRadius = 2.0


@dataclass(frozen=True)
class Duse:
    name: str
    pos: Vec3
    kraft: str  # "u" = unten, "s" = seite

# Auswahl des Punktes, auf den Verschoben werden soll
# Koordinatensystem-Konvention:
# x y z 
def bestimme_zielpunkt(
    kipp_in: str,
    schwerpunkt_rot: Vec3,
) -> np.ndarray:    
    # Um die Position der rechte hintere untere Ecke zu bestimmen wird der Abstand zwischen der Kippdüsen und des Schwerpunktes berechnet. U2/6/10/14/18 (-25, 0 x) U9/10/11/12 (0, 0, 40)
    zielpunkte_pro_kipp: Dict[str, Vec3] = {
        "o_x_p": (0, 0.0, 0.0),
        "o_x_n": (0, 0.0, 0.0),

        "u_x_p": (-25+abs(schwerpunkt_rot[0]), 0.0, 0.0),
        "u_x_n": (-25+abs(schwerpunkt_rot[0]), 0.0, 0.0),

        "u_z_p": (0.0, 0.0, 40-schwerpunkt_rot[2]),
        "u_z_n": (0.0, 0.0, 40.0-schwerpunkt_rot[2]),
    }
    
    return np.array(zielpunkte_pro_kipp[kipp_in], dtype=float) # Gibt die richtigen Koordinaten des Punktes zurück


# Dusen "Datenbank"
# Zugriff z.B.: print(duesen["DuseU1"].pos[0])
duesen: Dict[str, Duse] = {   #   x    y    z            #Die anderen Duesen werden nicht mehr benötigt
    #"DuseU1": Duse("DuseU1", (-10.0, 0.0, 10.0), "u"),
    "DuseU2": Duse("DuseU2", (-25.0, 0.0, 10.0), "u"),
    #"DuseU3": Duse("DuseU3", (-40.0, 0.0, 10.0), "u"),
    #"DuseU4": Duse("DuseU4", (-55.0, 0.0, 10.0), "u"),
    #"DuseU5": Duse("DuseU5", (-10.0, 0.0, 25.0), "u"),
    "DuseU6": Duse("DuseU6", (-25.0, 0.0, 25.0), "u"),
    #"DuseU7": Duse("DuseU7", (-40.0, 0.0, 25.0), "u"),
    #"DuseU8": Duse("DuseU8", (-55.0, 0.0, 25.0), "u"),
    "DuseU9": Duse("DuseU9", (-10.0, 0.0, 40.0), "u"),
    "DuseU10": Duse("DuseU10", (-25.0, 0.0, 40.0), "u"),
    "DuseU11": Duse("DuseU11", (-40.0, 0.0, 40.0), "u"),
    "DuseU12": Duse("DuseU12", (-55.0, 0.0, 40.0), "u"),
    #"DuseU13": Duse("DuseU13", (-10.0, 0.0, 55.0), "u"),
    "DuseU14": Duse("DuseU14", (-25.0, 0.0, 55.0), "u"),
    #"DuseU15": Duse("DuseU15", (-40.0, 0.0, 55.0), "u"),
    #"DuseU16": Duse("DuseU16", (-55.0, 0.0, 55.0), "u"),
    #"DuseU17": Duse("DuseU17", (-10.0, 0.0, 70.0), "u"),
    "DuseU18": Duse("DuseU18", (-25.0, 0.0, 70.0), "u"),
    #"DuseU19": Duse("DuseU19", (-40.0, 0.0, 70.0), "u"),
    #"DuseU20": Duse("DuseU20", (-55.0, 0.0, 70.0), "u"),


    "DuseO1": Duse("DuseO1", (-10.0, 10.0, 0.0), "o"),  # DuseO O ist keine Null sonder ein großes o
    "DuseO2": Duse("DuseO2", (-25.0, 10.0, 0.0), "o"),
    "DuseO3": Duse("DuseO3", (-40.0, 10.0, 0.0), "o"),
    "DuseO4": Duse("DuseO4", (-55.0, 10.0, 0.0), "o"),
    "DuseO5": Duse("DuseO5", (-10.0, 25.0, 0.0), "o"),
    "DuseO6": Duse("DuseO6", (-25.0, 25.0, 0.0), "o"),
    "DuseO7": Duse("DuseO7", (-40.0, 25.0, 0.0), "o"),
    "DuseO8": Duse("DuseO8", (-55.0, 25.0, 0.0), "o"),
    "DuseO9": Duse("DuseO9", (-10.0, 40.0, 0.0), "o"),
    "DuseO10": Duse("DuseO10", (-25.0, 40.0, 0.0), "o"),
    "DuseO11": Duse("DuseO11", (-40.0, 40.0, 0.0), "o"),
    "DuseO12": Duse("DuseO12", (-55.0, 40.0, 0.0), "o"),
    "DuseO13": Duse("DuseO13", (-10.0, 55.0, 0.0), "o"),
    "DuseO14": Duse("DuseO14", (-25.0, 55.0, 0.0), "o"),
    "DuseO15": Duse("DuseO15", (-40.0, 55.0, 0.0), "o"),
    "DuseO16": Duse("DuseO16", (-55.0, 55.0, 0.0), "o"),
    "DuseO17": Duse("DuseO17", (-10.0, 70.0, 0.0), "o"),
    "DuseO18": Duse("DuseO18", (-25.0, 70.0, 0.0), "o"),
    "DuseO19": Duse("DuseO19", (-40.0, 70.0, 0.0), "o"),
    "DuseO20": Duse("DuseO20", (-55.0, 70.0, 0.0), "o"),
}


def load_mesh(path: str) -> trimesh.Trimesh: #Lädt das Trimesh und gibt es aus
    
    #Laedt eine Mesh Datei STL
    obj = trimesh.load(path, force="mesh")
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

def kipp_in_to_rot(kipp_in: str) -> np.ndarray | None: #Rotation um den Schwerpunkt2 Gibt die Rotation zurück
    if kipp_in == "o_x_p":
        return rot_x_90k(1)
    if kipp_in == "o_x_n":
        return rot_x_90k(3)
    if kipp_in == "o_y_p":
        return rot_y_90k(1)
    if kipp_in == "o_y_n":
        return rot_y_90k(3)
    if kipp_in == "u_x_p":
        return rot_x_90k(1)
    if kipp_in == "u_x_n":
        return rot_x_90k(3)
    if kipp_in == "u_z_p":
        return rot_z_90k(1)
    if kipp_in == "u_z_n":
        return rot_z_90k(3)
    return None

def positionieren24() -> list[tuple[np.ndarray, str]]: #Erstellt die Position
    
    #Liefert 24 Rotationen in einer einfachen, logischen Reihenfolge:
    # Position 1..4: nur z-Drehungen (0/90/180/270)
    # Position 5..8: naechste "Grundlage" + wieder 0/90/180/270 um z
    

    I = np.eye(3)  # Einheitsmatrix wird erstellt

    # 6 Grundlagen = welche Seite unten ist (vereinfachtes, gut verstaendliches Set)
    bases = [    # Liste mit Tupeln     alle 6 Flächen liegen einmal unten Ausgangsfläche, die im nächsten Schritt dann um die zAchse gedreht werden. So erhalte ich alle Positionen
        ("x0", I),              # Basis 1: wie geladen
        ("x180", rot_x_90k(2)), # Basis 2: auf den Kopf 180 um x
        ("y90", rot_y_90k(1)),  # Basis 3: um y kippen
        ("y270", rot_y_90k(3)), # Basis 4: um y anders kippen
        ("x90", rot_x_90k(1)),  # Basis 5: um x kippen
        ("x270", rot_x_90k(3)), # Basis 6: um x anders kippen
    ]

    positions = []
    for base_label, base in bases:
        for k in range(4):
            # erst  Basis Kippen, dann Drehung um die Welt z Achse
            z_deg = k * 90 #Für das label
            label = f"{base_label}_z{z_deg}" #Fürd das label
            positions.append((rot_z_90k(k) @ base, label)) #Eigentliches Drehen

    return positions


# reale Kanten und Ecken werden bestimmt und Datenstruktur wird angelegt
# OPTION B START: axis parallel edge filtering
#Prüft ob es sich um eine echte geometrische Kante handelt, oder um eine vom Mesh erstellte Kante. Dies tritt z.B bei Löchern auf. Diese extra Kanten der Lochöffnung werden herausgerechnet, da disee nicht achsenparallel sind. Diese Lösung funktioniert nur bei Körpern, wo alle Kanten Achsenparallel verlaufen.
def axis_richtung(vec: np.ndarray, rel_tol: float, xyz_tol: float) -> int | None: #Prüfst, ob ein Richtungsvektor nahezu achsenparallel ist, indem die größte Komponente als Hauptachse ausgewählt wird  und alle anderen Komponenten nur innerhalb einer Toleranz zulässt sind Wenn das passt, gibst  die Achse (0=x, 1=y, 2=z) zurück, sonst None.
    xyz_v = np.abs(vec) # Nimmt den Betrag von x y z des Vektors ohne Vorzeichen 
    max_komp = float(np.max(xyz_v)) #Bestimmt die größte Kompnente also z.b x
    #if max_comp <= abs_tol: #Wenn der Wert null ist, ist keine Richtung definierrbar 
       # return None # Dann bricht die funktion ab, da es keinen gültigen Achsenvektor gibt 
    axis = int(np.argmax(xyz_v)) #Die Achse mit der größten Komponente wird als Kandidat gewählt zum bsp x 
    tol = max(xyz_tol, rel_tol * max_komp) #toleranz für Nebenachsen, da stl Dateien oft kleine Rundungsfehler haben für y z
    for i in range(3):
        if i != axis and xyz_v[i] > tol:#Wenn der Wert auf einer Nebenachse größer ist als die Toleranz non   Wenn y<x dann non oder wenn z<x dann non, da x die Hautpachse ist
            return None
    return axis
# OPTION B END

#Berechnen die echten Kanten dadurch, dass es nur Kanten als echt befindet die 90° zu einer anderen Kante stehen
def echte_kanten_und_ecken(
    mesh: trimesh.Trimesh,
    angle_deg: float = 90.0, #Zielwinkel
    tol_deg: float = 1.0,
    axis_rel_tol: float = 1e-3,
    axis_xyz_tol: float = 1e-5,
    min_corner_axes: int = 3, #Wie viele Achsenrichtungen an einer Ecke zusammenkommen müssen. Für eine Ecke 3
) -> dict:
    fa_edges = mesh.face_adjacency_edges #Hohlt alle Kanten, die zur benachtbarten Fläche gehören
    fa_angles = mesh.face_adjacency_angles #Fragt die Winekl zwischen diesen benachbarten Flächen ab
    if fa_edges is None or len(fa_edges) == 0: #Falls es keine Nachbarkante gibt, bricht es ab
        return {"edge_indices": np.empty((0, 2), dtype=int), "corner_indices": np.array([], dtype=int),
                "edge_coords": [], "corner_coords": []} #Gibt leeres array aus
    diff = np.abs(np.degrees(fa_angles) - angle_deg) #Berechen den Abstand des Flächenwinkel zum Zielwinkel 90
    edge90 = fa_edges[diff <= tol_deg] #Filter nur Kanten, deren Winkel im Toleranzbereich liegen.
    boundary = mesh.edges_boundary if hasattr(mesh, "edges_boundary") else np.empty((0, 2), dtype=int) #Fragt Randkanten des Meshs ab, da sie keine Nachbarfläche haben und damit keinen Winkelwert face_adjacency_angles
    if boundary is not None and len(boundary) > 0: #Prüft ob es Randkanten gibt
        edge90 = np.vstack([edge90, boundary]) #Fügt die Rankanten zu edge90 hinzu
    if len(edge90) == 0:
        return {"edge_indices": np.empty((0, 2), dtype=int), "corner_indices": np.array([], dtype=int),
                "edge_coords": [], "corner_coords": []} #Wenn es keine edge90, also 90°Kante,  gibt bircht die Funktion ab.
    edge90 = np.unique(np.sort(edge90, axis=1), axis=0)

    # OPTION B START: axis-parallel edge filtering
    #Hier wereden nur die achsenparallelen Kanten aus den 90°Kanten edge90 gefiltert
    axis_edges = [] #Liste für alle achsenparallelen Kanten
    axis_dirs = [] #Speicher, in welche Richtung jede gefundene Kante zeigt (0/1/2) x y z 
    for a, b in edge90: #Schleife ür alle 90° Kanten
        axis = axis_richtung(mesh.vertices[b] - mesh.vertices[a], axis_rel_tol, axis_xyz_tol) #Berechnet den Richtungsvektor der Kante und schut, ob er achsenparallel ist
        if axis is not None: #Wenn die achsenparallel
            axis_edges.append((a, b)) #Kanten hinzufügen
            axis_dirs.append(axis) # Achsenrichtung merken
    if len(axis_edges) == 0: #Falls keine achsenparallelen Kanten gefunden werden, wird ein leeres Ergebnis ausgegeben
        return {"edge_indices": np.empty((0, 2), dtype=int), "corner_indices": np.array([], dtype=int),
                "edge_coords": [], "corner_coords": []}

    axis_edges = np.array(axis_edges, dtype=int) #Wandelt die Kantenliste in ein NumPy Array um 
    axes_by_vertex = [set() for _ in range(len(mesh.vertices))] #Für jeden Eckpunt/Vertex wird ein Set erstellt, in das die Achsenrichtungen der Kanten eingetragen werden, die an diesem Eckpunkt/Vertex hängen
    for (a, b), axis in zip(axis_edges, axis_dirs): #Schleife über Kanten und Achsen 
        axes_by_vertex[int(a)].add(axis) #Achse bei Start-Vertex/Eckpunkt eintragen
        axes_by_vertex[int(b)].add(axis)#Achse bei End-Vertex/Eckpunkt eintragen
        #Das brauche ich, da ich ja später abfrage wie viel Achsen an einem Eckpunkt/vertex zusammenkommen. Wenn es mehr als 3 sind ist es ein Eckpunkt
    corner_idx = np.array( #Abfrage ob min_corner_axes also 3 erreicht ist. Wenn das so ist wird wird der Vertex_idx in corner _idx aufgenommen
        [i for i, axes in enumerate(axes_by_vertex) if len(axes) >= min_corner_axes],
        dtype=int,
    )
    edge_coords = [(mesh.vertices[a], mesh.vertices[b]) for a, b in axis_edges] #Hier werden die Kantenkoordianten aus den Vertex/Eckpunkt Koordinaten gebaut
    # OPTION B END
    corner_coords = [mesh.vertices[i] for i in corner_idx] # HIer werden die Eckkordinaten erzeugt
    return { #Die Ergebnisse werden ins Dict zurückgegeben
        "edge_indices": axis_edges,
        "corner_indices": corner_idx,
        "edge_coords": edge_coords,
        "corner_coords": corner_coords,
    }


#Debug Diagnose des Meshs             Prüft, ob OptionB das Probelem mit den Löchern im Mesh gelöst hat
def diagnose_mesh(mesh: trimesh.Trimesh, angle_deg: float = 90.0, tol_deg: float = 1.0, round_decimals: int = 5) -> None:
    vertices = mesh.vertices
    faces = mesh.faces
    vertex_count = int(len(vertices))
    face_count = int(len(faces))
    unique_rounded = np.unique(np.round(vertices, round_decimals), axis=0)
    duplicate_vertices = vertex_count - int(len(unique_rounded))

    fa_edges = mesh.face_adjacency_edges
    fa_angles = mesh.face_adjacency_angles
    if fa_edges is None or len(fa_edges) == 0:
        sharp_edge_count = 0
    else:
        diff = np.abs(np.degrees(fa_angles) - angle_deg)
        sharp = fa_edges[diff <= tol_deg]
        sharp = np.unique(np.sort(sharp, axis=1), axis=0) if len(sharp) else np.empty((0, 2), dtype=int)
        sharp_edge_count = int(len(sharp))

    boundary_edges = mesh.edges_boundary if hasattr(mesh, "edges_boundary") else np.empty((0, 2), dtype=int)
    boundary_edge_count = int(len(boundary_edges)) if boundary_edges is not None else 0

    echte = echte_kanten_und_ecken(mesh, angle_deg=angle_deg, tol_deg=tol_deg)
    corner_count = int(len(echte["corner_indices"]))
    edge_count = int(len(echte["edge_indices"]))

    try:
        components = mesh.split(only_watertight=False)
        component_count = int(len(components))
    except Exception:
        component_count = -1

    print("Mesh Diagnose:")
    print(f"  Vertices: {vertex_count}")
    print(f"  Faces: {face_count}")
    print(f"  Duplicate vertices (rounded {round_decimals}): {duplicate_vertices}")
    print(f"  Watertight: {mesh.is_watertight}")
    print(f"  Winding consistent: {mesh.is_winding_consistent}")
    print(f"  Euler number: {mesh.euler_number}")
    print(f"  Sharp edges (@{angle_deg} deg +/- {tol_deg}): {sharp_edge_count}")
    print(f"  Boundary edges: {boundary_edge_count}")
    print(f"  Corner candidates: {corner_count}")
    print(f"  Edge candidates: {edge_count}")
    if component_count >= 0:
        print(f"  Connected components: {component_count}")
    else:
        print("  Connected components: n/a")



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
    
   # if kipp_in == "o_y_p": # Benötige ich nicht mehr
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
   # if kipp_in == "o_y_n": #Benötige ich nicht mehr
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
    
    
    


def find_kippachse(pos: dict, kipp_in: str) -> dict | None: #Die Kippkante wird anhand der echten Bauteilkanten bestimmt. Es kann auch eine neue Kippkanten anhand z.B. der kleinsten Punkte erstellt werden. Das ist notwendig wenn die Bauteilkante unterborchn wird.
    ecken = np.array(pos["ecken"], dtype=float)  #Alle Eckpunkte werden aus pos geholt und in eine Arry umgewandelt

    ecken = np.round(ecken, 5)
    ecken_list = [tuple(row) for row in ecken]
    min_x = float(np.min(ecken[:, 0]))  #Kleinster x Wert aller Ecken wird gesucht
    max_x = float(np.max(ecken[:, 0]))
    min_y = float(np.min(ecken[:, 1]))
    max_y = float(np.max(ecken[:, 1]))
    min_z = float(np.min(ecken[:, 2]))
    max_z = float(np.max(ecken[:, 2]))

    if kipp_in == "u_z_p":
        a = (min_x, min_y, min_z)
        b = (min_x, min_y, max_z)
    elif kipp_in == "u_z_n":
        a = (max_x, min_y, min_z)
        b = (max_x, min_y, max_z)
    elif kipp_in == "u_x_p":
        a = (min_x, min_y, max_z)
        b = (max_x, min_y, max_z)
    elif kipp_in == "u_x_n":
        a = (min_x, min_y, min_z)
        b = (max_x, min_y, min_z)
    elif kipp_in == "o_x_p":
        a = (min_x, min_y, max_z)
        b = (max_x, min_y, max_z)
    elif kipp_in == "o_x_n":
        a = (min_x, max_y, max_z)
        b = (max_x, max_y, max_z)
    elif kipp_in == "o_y_p":
        a = (max_x, min_y, max_z)
        b = (max_x, max_y, max_z)
    elif kipp_in == "o_y_n":
        a = (min_x, min_y, max_z)
        b = (min_x, max_y, max_z)
    else:
        return None

    a = tuple(np.round(a, 5))
    b = tuple(np.round(b, 5))
    a_idx = ecken_list.index(a) if a in ecken_list else None
    b_idx = ecken_list.index(b) if b in ecken_list else None

    return { #Neue Kippkante wird aus den Bedingungen für a unb b aus den jeweiligen Eckpunkten bestimmt
        "edge_index": -1,
        "point_indices": (a_idx, b_idx),
        "a": a,
        "b": b,
    }


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
    pfade = [
        #Path(r"G:\Andere Computer\Mein Computer (2)\Bachelorarbeit\Programmierung\Cad Modelle\Dasha Modelle\Qf4i.stl"),
        #Path(r"G:\Andere Computer\Mein Computer (2)\Bachelorarbeit\Programmierung\Cad Modelle\Dasha Modelle\Ql4i.stl"),
        #Path(r"G:\Andere Computer\Mein Computer (2)\Bachelorarbeit\Programmierung\Cad Modelle\Dasha Modelle\Rk2i.stl"),
        Path(r"C:\Users\micha\Desktop\Bachelorarbeit\Programmierung\Cad Modelle\Dasha Modelle\Qf4i.stl"),
        Path(r"C:\Users\micha\Desktop\Bachelorarbeit\Programmierung\Cad Modelle\Dasha Modelle\Ql4i.stl"),
        Path(r"C:\Users\micha\Desktop\Bachelorarbeit\Programmierung\Cad Modelle\Dasha Modelle\Rk2i.stl"),
    ]
    #csv_path = Path(r"G:\Andere Computer\Mein Computer (2)\Bachelorarbeit\Programmierung\Cad Modelle\Dasha Modelle\kippkraft_tabelle.csv")
    csv_path = Path(r"C:\Users\micha\Desktop\Bachelorarbeit\Programmierung\Cad Modelle\Dasha Modelle\kippkraft_tabelle.csv")

     #Pfad der cvs
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file: #Öffnet die Datei und sorgt dafür, dass sie am Ende wieder geschlossen wird
        writer = csv.writer(csv_file, delimiter=";") # erstellt einen writer
        writer.writerow([ #Schreibt eine Kopfzeile
            "Datei",
            "Ausgangsposition",
            "kipp_in",
            "Kippposition",
            "Duese",
            "Impulskraft_t0_05s",
            "Impulskraft_t0_10s",
            "Impulskraft_t0_20s",
            "Hoehenunterschied",
            "y-Koordinate des Schwerpunkts vor dem Kippen",
            "y-Koordinate des Schwerpunkt nach dem Kippen",
            "Impulskraft_alterHoehenunterschied_t0_05s",
            "Impulskraft_alterHoehenunterschied_t0_10s",
            "Impulskraft_alterHoehenunterschied_t0_20s",
            "alter Hoehenunterschied",
            "alte y-Koordinate des Schwerpunkts nach dem Kippen",
            "zielpunkt_x",
            "zielpunkt_y",
            "zielpunkt_z",
            "Werkstueckhoehe_Position",
           #"Impulskraft verkippt",
           #"Hoehenunterschied verkippt",
           #"y-Schwerpunkt verkippt",
           #"y-Schwerpunkt2 verkippt"
        ])

        for pfad_teil in pfade: #Schleift durch alle Pfade
            print("STL-Pfad:", pfad_teil)
            print("Existiert die Datei?", pfad_teil.exists())

            # Mesh laden und ein paar Infos ausgeben
            m = load_mesh(str(pfad_teil))
            # Scale STL numbers from inch to mm.
            #m.apply_scale(1.0 / 25.4)
            # Flip Z to match MeshLab orientation (z up)
            flip_x = np.eye(4)
            flip_x[0, 0] = -1.0
            m.apply_transform(flip_x)
            print("Mesh geladen!")

            # Debug Ausgabe der Ecken und Kanten
            echte_geo = echte_kanten_und_ecken(m, angle_deg=90.0, tol_deg=1.0)  #print für die Konsole der Ausgangskoordinaten nicht transformiert
            print("Echte Ecken (Koordinaten):")
            for idx, p in enumerate(echte_geo["corner_coords"]):
                p_rounded = tuple(np.round(p, 5))
                print(f"  {idx}: {p_rounded}")
            print("Echte Kanten (Koordinaten):") #Debug Kanten
            for idx, ((a, b), (ia, ib)) in enumerate(zip(echte_geo["edge_coords"], echte_geo["edge_indices"])):
                a_rounded = tuple(np.round(a, 5))
                b_rounded = tuple(np.round(b, 5))
                print(f"  {idx}: {a_rounded} -> {b_rounded} | idx {ia}->{ib}")

            #Debug Ausgabe der Duesen 
            print("Duesen (Weltkoordinaten):") # print Koordinaten der Dusen
            for name in sorted(duesen.keys()):
                d = duesen[name]
                print(f"  {d.name}: {d.pos} ({d.kraft})")

            rotations = positionieren24()   # Liste mit 24 Matrizen (3x3) 

            positionen_koordinaten = []      #Liste für alle 24 Positionen wird erstellt 

            Schwerpunkt = np.array(m.center_mass, dtype=float) # Ausgangsschwerpunkt wird berechnet 
            schwerpunkt_h = np.array([*Schwerpunkt, 1.0], dtype=float)

            #Erstellt alle Meshs für alle Positionen

            for position_id, (R3, label) in enumerate(rotations, start=1):  #Transformation beginnt erst wird das Mesh pro Position orientiert dann verschoben
                T = np.eye(4)        # 4x4 Einheitsmatrix
                T[:3, :3] = R3       # 3x3 Rotationsmatrix damit kann das Mesh oder PUnkte rotiert/orientiert werden

                m_ver_orie = m.copy()     # Kopie, Original bleibt unverändert
                m_ver_orie.apply_transform(T)  #mesh wird durch T orientiert
                bounds_rot = m_ver_orie.bounds
                minx_rot, miny_rot, minz_rot = bounds_rot[0] #Untere Ecke der Bounding box 
                maxx_rot, maxy_rot, maxz_rot = bounds_rot[1]#Obere Ecke der Bounding box
                e5_rot = np.array([maxx_rot, miny_rot, minz_rot], dtype=float)#Baut den Punkt rechts hinten unten x = max, y = min, z = min   Diese Punkt soll verschoben werden Punkt (10,10,10)
                T_to_origin = np.eye(4)
                T_to_origin[:3, 3] = -e5_rot #Verschiebung um minus die Koordinaten des Punktes e5 (-10,-10,-10) T_shift verschiebt auf den Ursprung (0,0,0) 
                m_ver_orie.apply_transform(T_to_origin) #ganzes mesh wird durch T_shift verschoben
                 #m_ver_orie ist das verschobenen und orientierte Mesh
                T_origin = T_to_origin @ T
                schwerpunkt_rot = (T_origin @ schwerpunkt_h)[:3] #Der Schwerpunkt wird durch T_shift verschoben und duch T rotiert 

                echte =  echte_kanten_und_ecken(m_ver_orie, angle_deg=90.0, tol_deg=1.0) # transormierte echte Ecken und Kanten aus der Geometrie
                #echte_kanten_und_ecken gibt die Koordinaten der Ecken und Kanten soei die Ecken Indizese und die Kanten Indizes
                #echte enthält dann die orientierten Koordianten der Ecken und Kanten
                
                #Definierung des Dict
                positionen_koordinaten.append(  #Für jede Position wird ein Dictionary angelget mit den relevanten Daten
                    {  #benötige ich, wenn ich auf z.B Eckkordinaten einer bestimmten Position zugreifen will
                        "position_id": position_id,
                        "label": label,
                        "schwerpunkt": tuple(round(v, 5) for v in schwerpunkt_rot),
                        "ecken": echte["corner_coords"],
                        "kanten": echte["edge_indices"],
                        "kanten_coords": echte["edge_coords"],
                        "T_origin": T_origin,
                    }
                )
                

            diagnose_mesh(m, angle_deg=90.0, tol_deg=1.0, round_decimals=5)

            

            # Auswahl der Position
            # Position abfragen und Koordinatne der Ecken ausgeben 
            # Positionen automatisch durchlaufen
            for pos_id in range(1, len(positionen_koordinaten) + 1):



                pos = positionen_koordinaten[pos_id - 1] #Holt sich die Information aus dem Dictor über die Positionen. position_id label schwerpunkt ecken, kanten, kanten_coords
                schwerpunkt_pos = np.array(pos["schwerpunkt"], dtype=float)  #Transformierte Schwerpunktkoordinaten werden aus dem Dict geholt                
                pos2 = positionen_koordinaten[1]
                schwerpunkt_pos2 = pos2["schwerpunkt"]
                print("Schwerpunkt Position 2:", schwerpunkt_pos2)



                #Debug gibt transformierte Ecken und Kanten aus 
                print("Pos", pos["position_id"], "Ecken:") #Terminal Text
                for idx, e in enumerate(pos["ecken"]): #enumerate gibt idx und e aus
                    e_rounded = (
                        round(e[0], 5),
                        round(e[1], 5),
                        round(e[2], 5),
                    )
                    print(f"  {idx} {e_rounded}")  # Gibt den Index aus also z.b 0 und e_rounded also die gerundeten Eckenwerte aus
                print("Pos", pos["position_id"], "Kanten:")
                for idx, ((a, b), (ia, ib)) in enumerate(zip(pos["kanten_coords"], pos["kanten"])):#Gibt alle Transformierten Kanten + Index + den Index des Anfangs und des Endpunktes der Kante aus
                    a_rounded = tuple(np.round(a, 5))
                    b_rounded = tuple(np.round(b, 5))
                    print(f"  {idx}: {a_rounded} -> {b_rounded} | idx {ia}->{ib}")


                #Auswahl wie gekippt werden soll
                kipp_inputs = [  "u_x_n",  "u_z_n"]  #Um o_y_p und o_y_n "u_x_p" "u_z_p","o_x_p", "o_x_n", muss nicht mehr gekippt werden
                for kipp_in in kipp_inputs: #Schleife, die durch alle kipp_inputs schleift

                    # Mesh liegt je Position bereits auf (0,0,0), danach Verschiebung auf Zielpunkt.
                    T_pos_origin = pos["T_origin"] #Holt die gespeicherte Transfomrationsmatrix des rotierten und auf den Ursprung verschobenen Meshes
                    m_sel = m.copy() #Kopiert das Meshes
                    m_sel.apply_transform(T_pos_origin) #Bringt die Kopie in die T_origin position
                    bounds_sel = m_sel.bounds
                    minx, miny, minz = bounds_sel[0]
                    maxx, maxy, maxz = bounds_sel[1] #Stellt die kleinsten und größten Koordinaten des aktuellen Meshes bereit
                    werkstueckhoehe_position = float(maxy - miny)

                    y_min = m_sel.bounds[0][1] - 1.0 # Stellt die Startpunkte für den Strahlentest bereit.Immer 1 unter der Koordinate
                    z_min = m_sel.bounds[0][2] - 1.0

                    zielpunkt = bestimme_zielpunkt(kipp_in, tuple(schwerpunkt_pos)) #Wählt den richtigen Zielpunkt abhängig von kipp_in aus
                    T_shift = np.eye(4)
                    T_shift[:3, 3] = zielpunkt
                    T_total = T_shift @ T_pos_origin
                    m_sel.apply_transform(T_shift) #verschiebt das Mesh auf den Zielpunkt

                    schwerpunkt_pos = (T_total @ schwerpunkt_h)[:3] #Transformiert den Schwerpunkt

                    echte_kanten_und_ecken_transformiert = echte_kanten_und_ecken(m_sel, angle_deg=90.0, tol_deg=1.0) 
                    pos_transformiert = { #Die Koordinaten Ecken der echten_kanten_und_ecken werden auf den Zielpunkt verschoben.
                        "ecken": echte_kanten_und_ecken_transformiert["corner_coords"],
                    } #Die Koordinaten der verschobenen Ecken in pos_transfomiert brauche ich, um die richtige Kippachse zu finden find_kippachse()

                    nozzle_positions = []                 #Dusen-Koordinaten aus dict werden zu Kugeln
                    for d_marker in duesen.values():
                        p = np.array(d_marker.pos, dtype=float)
                        nozzle_positions.append(p)
                    nozzle_markers = create_nozzle_cylinders(nozzle_positions) #Erstellt Kugeln statt Zylinder
                    #out_dir = Path(r"G:\Andere Computer\Mein Computer (2)\Bachelorarbeit\Programmierung\Cad Modelle\Dasha Modelle\Modelle")
                    out_dir = Path(r"C:\Users\micha\Desktop\Bachelorarbeit\Programmierung\Cad Modelle\Dasha Modelle\Modelle")
                    out_dir.mkdir(parents=True, exist_ok=True)
                    base_name = pfad_teil.stem
                    m_export = trimesh.util.concatenate([m_sel, nozzle_markers]) #Exportier das verschobene Mesh und die Kugeln
                    out_path = out_dir / f"{base_name}_pos{pos['position_id']:02d}_{pos['label']}_{kipp_in}.stl"
                    m_export.export(out_path)
                    print("Exportiert:", out_path)


                    #Bestimmt welche Düse zum Kippen zu gebrauchen ist
                    funkduse = kipp_duse(kipp_in, tuple(schwerpunkt_pos)) #kipp_duse wird aufgerufen der schwerpunkt wird übergeben und funk_dusen wird zurückgegeben

                    print("Schwerpunkt transformiert:", schwerpunkt_pos)

                    print(f"kipp_duse {kipp_in} ( Schwerpunkt2):")

                    #Berechnung des Schwerpunkt2 
                    rot90 = kipp_in_to_rot(kipp_in) # Holt 90° Rotation die zu kipp_in gehoert
                    rot_ma_akt_pos, _ = rotations[pos_id - 1] #Holt die Rotationsmatrix der aktuellen Position
                    z_rot = rot90 @ rot_ma_akt_pos #Berechnet die Ziel Rotation, erst aktuelle Position und dann Kippposition
                    z_pos = None #Ziel Positon
                    z_nam = None #Zielname
                    for idx, (R, label) in enumerate(rotations, start=1): #Durchläuft alle 24 Position 
                        print("z_rot",z_rot)
                        print("R",R)
                        print("idx", idx)
                        print("label",label)
                        if np.allclose(R, z_rot, atol=1e-6): #Prüft, ob eine der gespeicherten Rotationen der Ziel Rotation entspricht
                            z_pos = idx     #mit dem Index kann dann später im Dict der Schwerpunkt mit der Position(Index)angefordert werden
                            z_nam = label
                            break
                    print("Schwerpunkt2 label: ", z_nam, "Schwerpunkt2 pos:", z_pos)
                    #Verschiebt und rotiert den Schwerpunkt2 erst auf den Ursprung und dann auf die Zielposition, dazu wird das mesh der neuen Position wie das Mesh der aktuellen Position verschoben rotiert und nocheinmal verschoben
                    R3_target, _ = rotations[z_pos - 1]
                    T_target = np.eye(4)
                    T_target[:3, :3] = R3_target
                    m_target = m.copy()
                    m_target.apply_transform(T_target)
                    bounds_target = m_target.bounds  
                    minx_t, miny_t, minz_t = bounds_target[0]
                    maxx_t, maxy_t, maxz_t = bounds_target[1]
                    e5_target = np.array([maxx_t, miny_t, minz_t], dtype=float)

                    T_shift_target = np.eye(4)
                    T_shift_target[:3, 3] = -e5_target
                    T_origin_target = T_shift_target @ T_target

                    T_to_ziel_target = np.eye(4)
                    T_to_ziel_target[:3, 3] = zielpunkt
                    T_total_target = T_to_ziel_target @ T_origin_target
                    schwerpunkt_pos2 = (T_total_target @ schwerpunkt_h)[:3]
                    print("Koordinaten des Schwerpunkts2: ", tuple(np.round(schwerpunkt_pos2, 5)))
                    print("Schwerpunkt1 zum Vergleich:", schwerpunkt_pos) #Debug



                    #Berechnung der finalen Duesen
                    finaldusen = []
                    for d, dx, dy, dz in funkduse:
                        print(f"  {d.name}: {d.pos} | dx={dx} dy={dy} dz={dz}")

                        #Strahltest, ob ein Strahl von der Düse aus kommend auf die Oberfläche des Meshs trifft
                        if d.kraft == "u":
                            origin = np.array([d.pos[0], y_min, d.pos[2]], dtype=float)  #Setzt den Startpunkt des Strahls gleich x z der Düse
                            direction = np.array([0.0, 1.0, 0.0], dtype=float) #Richtung, in die der Strahl zeigt
                            locations, _, _ = m_sel.ray.intersects_location([origin], [direction]) #Berechnet alle Schnittpunkte des Strahls mit der Mesh Oberfläche
                            surface_match = len(locations) > 0   #Wenn es einen Treffer gab, wird true in surface_match abgespeichert
                            print("    xz_surface:", surface_match)
                            if surface_match:  #Wenn true dann
                                print("    xz_locations:", locations) #Debug gibt alle Schnittpunkt aus
                                distances = np.linalg.norm(locations - origin, axis=1) # Berechnet den Abstand vom Strahl Startpunkt zu den Schnittpunkten
                                closest = locations[np.argmin(distances)] #Wählt den Schnittpunkt aus, der am nächsten am Strahl Startpunkt liegt
                                finaldusen.append({"duse": d, "dx": dx, "dy": dy, "dz": dz, "surface": closest.tolist()}) 

                        elif d.kraft == "o":
                            origin = np.array([d.pos[0], d.pos[1], z_min], dtype=float)
                            direction = np.array([0.0, 0.0, 1.0], dtype=float)
                            locations, _, _ = m_sel.ray.intersects_location([origin], [direction])
                            surface_match = len(locations) > 0
                            print("    xy_surface:", surface_match)
                            if surface_match:
                                print("    xy_locations:", locations)
                                distances = np.linalg.norm(locations - origin, axis=1)
                                closest = locations[np.argmin(distances)]
                                finaldusen.append({"duse": d, "dx": dx, "dy": dy, "dz": dz, "surface": closest.tolist()})

                    print("finaldusen (surface=True):")   #Finale Ausgabe, der passenden Duesen
                    for eintrag in finaldusen:
                        d = eintrag["duse"]
                        print(f"  {d.name}: {d.pos} | dx={eintrag['dx']} dy={eintrag['dy']} dz={eintrag['dz']} surface={eintrag['surface']}")

                    if not finaldusen:
                        print("Finale Duese: None")
                        continue #Wenn finale duse bei einer Position und kipp_in leer ist, dann abruch und nächste kipp_in

                    for finalduse in finaldusen: #Schleife durch alle möglichen Düsen
                        print("Finale Duese: ", finalduse)


                        #Berechnung der Kippachse
                        finalkippachse = find_kippachse(pos_transformiert, kipp_in)  #Kippachse wird in der Funktion bestimmt alle Koordianten werden von pos_transformiert sowie die einegabe kipp_in übergeben
                        if finalkippachse is not None:
                            print("Kippkante: ",finalkippachse)
                        else:
                            print("Kippachse: keine passende Kante gefunden")
                            continue


                    #Berechnung von OD
                        OD = None
                        ds = np.array(finalduse["surface"], dtype=float) # ds ist der Dusenschnittpunkt
                        ka = np.array(finalkippachse["a"], dtype=float) # Endpunkt a der Kippkante 
                        kb = np.array(finalkippachse["b"], dtype=float) # Endpunkt b der Kippkante
                        kab = kb - ka #Berechenet Richtungsvektor der Kante 

                        if kipp_in in {"u_x_p", "u_x_n"}:
                            OD = abs(ds[2] - ka[2])  #Bestimmt den z Unterschied zwischen finalenKippkante und finalenDuse
                        elif kipp_in in {"u_z_p", "u_z_n"}:
                            OD = abs(ds[0] - ka[0])  #Bestimmt den x Unterschied zwischen finalenKipkante und der finalenDuse
                        elif kipp_in in {"o_x_p", "o_x_n"}: #OD ist der y Unterschied der finalenKippkante und der finalenDuse
                            OD = abs(ds[1] - ka[1])

                        print("Abstand zwischen O und D: ", OD)
                        if OD is None:
                            continue

                        #Berechnung des Maassenträgheitsmoments  (Mesh muss geschlossen sein)
                        rho = 1.287e-6  # kg/mm^3
                        I_kipp = None

                        kante_len1 = np.linalg.norm(kab) #Kantenlänge
                        e_k_ab = kab / kante_len1 #Normiert Richtungsvektor der Achse. nkab Einheits-Richtungsvektor
                        m_eig = m_sel.mass_properties #Zieht die Masseeigenschaften aus dem Mesh (Volumen, Traegheit, Schwerpunkt)
                        tm = m_eig["inertia"] * rho #Traeheitsmatrix um den Schwerpunkt mit der Dichte skaliert
                        mass = m_eig["volume"] * rho #Masse

                        schwerpunkt_ka = schwerpunkt_pos - ka #Vektor vom Achsenpunkt ka zum Schwerpunkt
                        schwerpunkt_ka_senk = schwerpunkt_ka - np.dot(schwerpunkt_ka, e_k_ab) * e_k_ab #Entfernt den Anteil von schwerpunkt_ka der in Achsrichtung zeigt, gibt den senkrechten Anteil (Abstand zur Achse) zurück
                        schwerpunkt_ka_senk2 = np.dot(schwerpunkt_ka_senk, schwerpunkt_ka_senk)  #Quadrat des Abstands zwischen Schwerpunkt und Achse
                        Iachse = float(e_k_ab @ tm @ e_k_ab) #Trägheitsmoment um die Achse durch den Schwerpunkt
                        I_kipp = Iachse + mass * schwerpunkt_ka_senk2 #Trägheitsmoment um die Kippachse
                        print("Massentraegheitsmoment um Kippachse: ", I_kipp)




                        #Berechnung des alten Höhenunterschieds
                        h_alt = abs(round(schwerpunkt_pos2[1] - schwerpunkt_pos[1],5)) #Höhenunterschied des Schwerpunktes vor und nach dem Kippen. Funktioniert nicht für o_x_p und o_y_n
                        

                        #Neue Berechnung des Höhenunterschieds
                        schwerpunkt_kippachse_abstand = float(np.linalg.norm(schwerpunkt_ka_senk)) #schwerpunkt_ka_senk ist der senkrechte Vektor vom Schwerpunkt zur Kippachse, diesen hab ich schon berechnet. Np.linalg.norm() berechnet dann die Länge des Vektors. So habe ich die Länge vom Schwerpunkt zur Kippachse.
                        #Der Abstand zwischen schwerpunkt und kippachse ist somit der neue y-Wert für Schwerpunkt2 und damit ist der Höhenuntschied:
                        h=abs(round(float(schwerpunkt_kippachse_abstand) - float(schwerpunkt_pos[1]),5)) # abs() um ein negatives h zu vermeiden, da sonst die Wurzel negativ würde
                        print("Höhenunterschied:",h)

                        alpha = math.radians(20)
                        beta = math.radians(10) 
                        h_schwerpunkt_kippachse_abstand = schwerpunkt_kippachse_abstand *math.cos(alpha)* math.cos(beta)
                        h_schwerpunkt_pos = schwerpunkt_pos[1] * math.cos(alpha)* math.cos(beta)
                        h_verkippt = abs(round(float(h_schwerpunkt_kippachse_abstand)-float( h_schwerpunkt_pos),5))
                        print("Verkippter Höhenunterschied:",h_verkippt)
                        

                        #Berechnung der Impulskraft F fuer drei verschiedene Zeiten
                        t_werte = [0.05, 0.10, 0.20] #s
                        g = 9810 #mm/s2

                        #Debug
                        print("Masse:",mass)
                        print("g:",g)
                        impulskraefte = {}
                        impulskraefte_alt = {}
                        for t in t_werte:
                            F_t = math.sqrt((mass * g * h * 2 * math.pow(I_kipp, 2))  / (I_kipp * math.pow(t, 2) * math.pow(OD, 2))) # F in kg/mms2
                            F_t = round(F_t / 1000, 5)  # F in kg/ms2 (N)
                            impulskraefte[t] = F_t
                            print(f"Kraft bei t={t:.2f}s: ", F_t, ("N"))

                            #Berechnung der Impulskraft mit dem alten Höhenunterschied für drei Zeiten

                            F_alt_t = math.sqrt((mass * g * h_alt * 2 * math.pow(I_kipp, 2))  / (I_kipp * math.pow(t, 2) * math.pow(OD, 2))) # F_alt in kg/mms2
                            F_alt_t = round(F_alt_t / 1000, 5)  # F in kg/ms2 (N)
                            impulskraefte_alt[t] = F_alt_t
                            print(f"Kraft_alt bei t={t:.2f}s: ", F_alt_t, ("N"))

                        #Berechnung verkippte Impulskraft
                        #F_verkippt = math.sqrt((mass * g * h_verkippt * 2 * math.pow(I_kipp, 2))  / (I_kipp * math.pow(t, 2) * math.pow(OD, 2))) # F in kg/mms2
                        #F_verkippt = round(F_verkippt / 1000, 5)
                        #print("Verkippte Kraft: ",F_verkippt,("N"))

                        writer.writerow([ #schreibt eine neue Zeile in die csv 
                            pfad_teil.name,
                            pos_id,
                            kipp_in,
                            z_pos,
                            finalduse["duse"].name,
                            f"{impulskraefte[0.05]:.5f}".replace(".", ","),
                            f"{impulskraefte[0.10]:.5f}".replace(".", ","),
                            f"{impulskraefte[0.20]:.5f}".replace(".", ","),
                            f"{h:.5f}".replace(".", ","),
                            f"{schwerpunkt_pos[1]:.5f}".replace(".", ","),
                            f"{schwerpunkt_kippachse_abstand:.5f}".replace(".", ","),
                            f"{impulskraefte_alt[0.05]:.5f}".replace(".", ","),
                            f"{impulskraefte_alt[0.10]:.5f}".replace(".", ","),
                            f"{impulskraefte_alt[0.20]:.5f}".replace(".", ","),
                            f"{h_alt:.5f}".replace(".", ","),
                            f"{schwerpunkt_pos2[1]:.5f}".replace(".", ","),
                            f"{zielpunkt[0]:.5f}".replace(".", ","),
                            f"{zielpunkt[1]:.5f}".replace(".", ","),
                            f"{zielpunkt[2]:.5f}".replace(".", ","),
                            f"{werkstueckhoehe_position:.5f}".replace(".", ","),
                            #f"{F_verkippt:.5f}".replace(".", ","),
                            #f"{h_verkippt:.5f}".replace(".", ","),
                            #f"{h_schwerpunkt_pos:.5f}".replace(".", ","),
                            #f"{h_schwerpunkt_kippachse_abstand:.5f}".replace(".", ","),
                            
                
                


                        ])

    print("CSV geschrieben:", csv_path)




