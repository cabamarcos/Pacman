"""Render measured benchmark results and training curves as dependency-free SVG."""
import csv
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
directory = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "reference"
results = json.loads((directory / "benchmark.json").read_text(encoding="utf-8"))
summary = results["summary"]
names = {"random": "Aleatorio", "untrained_q": "Q sin entrenar", "q_learning": "Q-learning", "greedy_bfs": "Rutas con BFS"}
colors = {"random": "#93a3b8", "untrained_q": "#b0bfce", "q_learning": "#ffe66b", "greedy_bfs": "#67dfc5"}


def svg_header(width, height):
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><rect width="100%" height="100%" rx="20" fill="#101c2c"/><g font-family="Arial, sans-serif" fill="#e6edf5">'


def label(x, y, text, size=14, fill=None):
    return f'<text x="{x}" y="{y}" font-size="{size}"'+(f' fill="{fill}"' if fill else '')+f'>{text}</text>'


pieces = [svg_header(880, 354), label(30, 42, "Evaluación · dos fantasmas estáticos, tres mapas conocidos", 20),
          label(30, 68, "270 partidas por agente · 3 semillas de entrenamiento · límite: 120 movimientos", 13, "#a0b0c5"),
          label(235, 106, "Partidas completadas", 14), label(575, 106, "Movimientos medios¹", 14)]
for index, (key, row) in enumerate(summary.items()):
    y = 137 + index * 44
    pieces += [label(30, y+10, names[key]),
               f'<rect x="235" y="{y-4}" width="230" height="20" rx="5" fill="#26374b"/>',
               f'<rect x="235" y="{y-4}" width="{230*row["win_rate"]:.2f}" height="20" rx="5" fill="{colors[key]}"/>',
               label(480, y+11, f'{100*row["win_rate"]:.1f}%'),
               f'<rect x="575" y="{y-4}" width="{160*row["mean_moves"]/120:.2f}" height="20" rx="5" fill="{colors[key]}"/>',
               label(750, y+11, f'{row["mean_moves"]:.1f}')]
pieces += [label(30, 326, "¹ Incluye los intentos fallidos, que consumen 120 movimientos. Semillas de evaluación separadas.", 12, "#a0b0c5"), '</g></svg>']
(directory / "comparison.svg").write_text("".join(pieces), encoding="utf-8")

with (directory / "training.csv").open(encoding="utf-8") as stream:
    training = list(csv.DictReader(stream))
episodes = results["protocol"]["episodes_per_run"]
window = 100
pieces = [svg_header(880, 380), label(30, 40, "Aprendizaje durante el entrenamiento", 21),
          label(30, 66, "Media de movimientos por bloques de 100 episodios · exploración ε: 0,8 → 0,05", 13, "#a0b0c5")]
left, top, width, height = 65, 100, 760, 220
for value in (0, 30, 60, 90, 120):
    y = top + height * (1-value/120)
    pieces += [f'<path d="M{left} {y}h{width}" stroke="#304056"/>', label(26, y+5, str(value), 12)]
for index, seed in enumerate(results["protocol"]["run_seeds"]):
    color = ("#ffe66b", "#67dfc5", "#8cbaff")[index]
    rows = [row for row in training if int(row["run_seed"]) == seed]
    points = []
    for offset in range(0, len(rows), window):
        chunk = rows[offset:offset+window]
        x = left + width * (offset + len(chunk)/2) / episodes
        y = top + height * (1-statistics.mean(float(r["moves"]) for r in chunk)/120)
        points.append(f"{x:.2f},{y:.2f}")
    pieces += [f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2.5"/>',
               label(475+index*120, 90, f"Semilla {seed}", 12, color)]
for tick in range(0, episodes+1, 500):
    pieces.append(label(left+width*tick/episodes-10, 343, str(tick), 12))
pieces += [label(65, 365, "Episodios · curva de entrenamiento, no validación en mapas nuevos", 12, "#a0b0c5"), '</g></svg>']
(directory / "training.svg").write_text("".join(pieces), encoding="utf-8")
print(f"Generated comparison.svg and training.svg in {directory}")
