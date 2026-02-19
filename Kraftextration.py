from pathlib import Path

# =========================
# HIER DEINEN "LINK" (PFAD) EINTRAGEN
# Beispiele:
# Windows: r"C:\Messdaten\Kraft"
# Linux/Mac: "/home/user/messdaten/kraft"
# Netzwerk: r"\\SERVER\Freigabe\Messdaten"
# =========================
DATA_DIR = Path(r"G:\Andere Computer\Mein Computer (2)\Bachelorarbeit\Programmierung\Cad Modelle\Dasha Modelle\Versuchsdaten (1)\Kraftmessdose\1")

EXPECTED_STOESSE = 10      # es gab 10 Kraftstöße
BIN_SIZE_S = 1.0           # Abstand 1 Sekunde
START_AT_FIRST_PEAK = True # ignoriert lange 0-Abschnitte am Anfang
THRESHOLD = 0.0            # wenn Rauschen da ist, z.B. 0.05 setzen


def parse_time_to_seconds(t: str) -> float:
    # unterstützt z.B. "24:29,3" oder "00:00:01,200"
    t = t.strip().replace(",", ".")
    parts = t.split(":")
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    # len == 3
    h, m, s = parts
    return int(h) * 3600 + int(m) * 60 + float(s)


def read_file_max_per_second_bins(path: Path) -> dict[int, float]:
    # dict: bin_index -> max_value
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(lines) < 2:
        return {}

    # erste Zeile ist Header
    data_lines = lines[1:]

    times = []
    vals = []
    for line in data_lines:
        line = line.strip()
        if not line:
            continue

        # meistens TAB-getrennt
        if "\t" in line:
            parts = line.split("\t")
        else:
            parts = line.split()  # fallback

        if len(parts) < 2:
            continue

        try:
            t = parse_time_to_seconds(parts[0])
            v = float(parts[1].replace(",", "."))
        except Exception:
            continue

        times.append(t)
        vals.append(v)

    if not times:
        return {}

    t0 = min(times)

    max_by_bin: dict[int, float] = {}
    for t, v in zip(times, vals):
        b = int((t - t0) // BIN_SIZE_S)
        if (b not in max_by_bin) or (v > max_by_bin[b]):
            max_by_bin[b] = v

    return max_by_bin


def main():
    files = sorted(DATA_DIR.glob("*.txt"))
    if not files:
        print(f"Keine .txt Dateien gefunden in: {DATA_DIR}")
        return

    for f in files:
        max_by_bin = read_file_max_per_second_bins(f)
        if not max_by_bin:
            print(f"{f.name}: keine Daten lesbar")
            continue

        # Start-Bin finden (erstes Bin, dessen Max > Threshold), damit 0-Vorlauf ignoriert wird
        start_bin = 0
        if START_AT_FIRST_PEAK:
            bins_over = sorted([b for b, m in max_by_bin.items() if m > THRESHOLD])
            if bins_over:
                start_bin = bins_over[0]

        stosswerte = []
        for i in range(EXPECTED_STOESSE):
            b = start_bin + i
            m = max_by_bin.get(b, None)
            if m is not None:
                stosswerte.append(m)

        if stosswerte:
            mittelwert = sum(stosswerte) / len(stosswerte)
            print(f"{f.name}: {mittelwert:.4f}")
        else:
            print(f"{f.name}: Mittelwert nicht berechenbar")

if __name__ == "__main__":
    main()
