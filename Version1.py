from dataclasses import dataclass
from typing import Dict, Tuple

Vec3 = Tuple[float, float, float]  # (x, y, z) in Metern


@dataclass(frozen=True)
class Düse:
    name: str
    pos: Vec3
    kraft: str



# x = vorne, y = links, z = oben
Ursprung: Vec3 = (0.0, 0.0, 0.0)


# 2) Düsen "Datenbank" print(düsen["DüsenU1"].pos[0])      "u" steht dafür, dass die Kraft von unten wirkt        "s" Kraft von der Seite
düsen: Dict[str, Düse] = {   # x      y     z
    "DüseU1": Düse("DüseU1", (10, 10, 0),"u"),
    "DüseU2": Düse("DüseU2", (10,  20, 0),"u"),
    "DüseU3": Düse("DüseU3", (10,  30, 0),"u"),
    "DüseU4": Düse("DüseU4", (20, 10, 0),"u"),
    "DüseU5": Düse("DüseU5", (20,  20, 0),"u"),
    "DüseU6": Düse("DüseU6", (20,  30, 0),"u"),
    "DüseU7": Düse("DüseU7", (30, 10, 0),"u"),
    "DüseU8": Düse("DüseU8", (30,  20, 0),"u"),
    
    "DüseO1":   Düse("DüseO1",   (0, 10, 10),"s"),
    "DüseO2":   Düse("DüseO2",   (0, 20, 10),"s"),
    "DüseO3":   Düse("DüseO3",   (0, 30, 10),"s"),
    "DüseO4":   Düse("DüseO4",   (0, 10, 20),"s"),
    "DüseO5":   Düse("DüseO5",   (0, 20, 20),"s"),
    "DüseO6":   Düse("DüseO6",   (0, 30, 20),"s"),
    "DüseO7":   Düse("DüseO7",   (0, 10, 30),"s"),
    "DüseO8":   Düse("DüseO8",   (0, 20, 30),"s"), 
}

#for name, d in düsen.items():
#    print(name, d.pos)
#print(Ursprung[1])
#print(düsen["DüseO5"].kraft)



import numpy as np
import trimesh

# ---------- 1) Laden: funktioniert sicher mit STL/OBJ/PLY ----------
def load_mesh(path: str) -> trimesh.Trimesh:
    obj = trimesh.load(path, force="mesh")

    # Manche Formate laden als "Scene" (mehrere Meshes). Dann zusammenfügen:
    if isinstance(obj, trimesh.Scene):
        obj = trimesh.util.concatenate(tuple(obj.geometry.values()))
        print("Läuft")

    if not isinstance(obj, trimesh.Trimesh):
        raise TypeError(f"Datei konnte nicht als Mesh geladen werden: {path}")

    return obj




# ---------- Beispiel: 3 Teile laden ----------
parts = {
    "teil1": load_mesh(r"C:\Users\micha\Desktop\Bachelorarbeit\Programmierung\Cad Modelle\Test-Bauteil1\teil1.stl"),
    #"teil2": load_mesh("teil2.stl"),
    #teil3": load_mesh("teil3.stl"),
}

# Wähle ein Teil
m = parts["teil1"].copy()  # copy(), damit das Original erhalten bleibt


# Visualisieren (öffnet ein Fenster)
scene = trimesh.Scene()
scene.add_geometry(m)
scene.show()
