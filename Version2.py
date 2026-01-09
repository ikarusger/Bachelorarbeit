from dataclasses import dataclass
from typing import Dict, Tuple
from pathlib import Path

import trimesh

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


if __name__ == "__main__":
    # >>> HIER: Pfad zu deiner STL-Datei eintragen (Dateiname muss exakt stimmen!)
    pfad_teil1 = Path(r"C:\Users\micha\Desktop\Bachelorarbeit\Programmierung\Cad Modelle\Test-Bauteil1.stl")

    print("STL-Pfad:", pfad_teil1)
    print("Existiert die Datei?", pfad_teil1.exists())

    if not pfad_teil1.exists():
        raise FileNotFoundError(f"STL-Datei nicht gefunden: {pfad_teil1}")

    # Mesh laden und ein paar Infos ausgeben
    m = load_mesh(str(pfad_teil1))
    print("Mesh geladen!")
    print("Vertices:", len(m.vertices))
    print("Faces:", len(m.faces))
    print("Bounds (min/max):", m.bounds)

    # Visualisieren
    scene = trimesh.Scene()
    scene.add_geometry(m)
    scene.show(smooth=False)