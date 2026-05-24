from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

RACINE = Path(__file__).resolve().parents[1] / "Reconstruction_3D_propre"
WEB = Path(__file__).resolve().parent
sys.path.insert(0, str(RACINE / "src"))

from reconstruction3d.chemins import DOSSIER_CAPTURES, LARGEUR_PROJECTEUR_PX  # noqa: E402
from reconstruction3d.franges import (  # noqa: E402
    binariser_franges,
    charger_captures,
    decoder_coordonnees_projecteur,
    masque_objet_depuis_franges,
)
from reconstruction3d.geometrie import ajuster_plan, filtrer_objet, filtrer_plan, passer_dans_repere_plan, trianguler  # noqa: E402
from reconstruction3d.masques import appliquer_masque, charger_masque  # noqa: E402
from reconstruction3d.pipeline import charger_matrices_calibration  # noqa: E402


OBJETS = [
    ("plan", "Plan", "plan.html", True),
    ("pyramide a plat", "Pyramide a plat", "pyramide-a-plat.html", True),
    ("Bras", "Bras", "bras.html", True),
    ("Sac Zainab", "Sac Zainab", "sac-zainab.html", True),
    ("Mines", "Mines", "mines.html", False),
    ("Caisse bricoleur", "Caisse bricoleur", "caisse-bricoleur.html", True),
]

MAX_POINTS_PAR_N = 38000
TOLERANCE_SOUS_PLAN_MM = 5.0
EPAISSEUR_PLAN_MM = 12.0


def nombre_captures_disponibles(dossier: Path, maximum: int) -> int:
    n = 0
    for i in range(1, maximum + 1):
        if (dossier / f"capture_{i}.png").exists():
            n = i
        else:
            break
    if n == 0:
        raise FileNotFoundError(dossier)
    return n


def reconstruire_points(nom_objet: str, n: int, mr: np.ndarray, me: np.ndarray) -> np.ndarray:
    dossier = DOSSIER_CAPTURES / nom_objet
    captures = charger_captures(dossier, n)
    masque, _ = charger_masque(dossier, captures[0].shape)
    captures = appliquer_masque(captures, masque)
    franges = binariser_franges(captures, masque)
    codes = decoder_coordonnees_projecteur(franges, masque)
    if n == 1:
        masque_objet = masque & (captures[0] > 35)
    else:
        masque_objet = masque_objet_depuis_franges(captures, masque)
    points = trianguler(codes, masque_objet, mr, me, LARGEUR_PROJECTEUR_PX / float(2**n))
    return filtrer_plan(points) if nom_objet.lower() == "plan" else filtrer_objet(points)


def sous_echantillonner(points: np.ndarray) -> np.ndarray:
    if len(points) <= MAX_POINTS_PAR_N:
        return points
    idx = np.linspace(0, len(points) - 1, MAX_POINTS_PAR_N, dtype=int)
    return points[idx]


def donnees_objet(nom_objet: str, avec_slider: bool, mr: np.ndarray, me: np.ndarray) -> dict:
    max_n = 8 if avec_slider else nombre_captures_disponibles(DOSSIER_CAPTURES / nom_objet, 8)
    donnees = {}
    for n in range(1, max_n + 1):
        points_plan = reconstruire_points("plan", n, mr, me)
        centre_plan, axes_plan, _ = ajuster_plan(points_plan)

        if nom_objet.lower() == "plan":
            points = passer_dans_repere_plan(points_plan, centre_plan, axes_plan)
        else:
            points_bruts = reconstruire_points(nom_objet, n, mr, me)
            points = passer_dans_repere_plan(points_bruts, centre_plan, axes_plan)
            points = points[points[:, 2] >= -TOLERANCE_SOUS_PLAN_MM]

        points = sous_echantillonner(points)
        donnees[str(n)] = np.round(points, 2).tolist()
    return donnees


def html_vue(titre: str, donnees: dict, avec_slider: bool, gradient_defaut: bool) -> str:
    n_defaut = max(int(k) for k in donnees)
    return f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{titre} - reconstruction 3D</title>
<style>
html,body{{margin:0;height:100%;overflow:hidden;background:#101114;color:#f2f2f2;font-family:Arial,sans-serif}}
canvas{{display:block;width:100vw;height:100vh;cursor:grab;background:#101114}}
canvas:active{{cursor:grabbing}}
#hud{{position:fixed;left:14px;top:12px;background:rgba(0,0,0,.58);border:1px solid rgba(255,255,255,.16);border-radius:8px;padding:10px 12px;font-size:13px;line-height:1.35;box-shadow:0 2px 10px rgba(0,0,0,.28)}}
#controls{{display:grid;gap:8px;margin-top:9px;min-width:260px}}
label{{display:flex;align-items:center;gap:8px;white-space:nowrap}}
input[type=range]{{width:160px}}
button{{border:1px solid rgba(255,255,255,.24);background:#202226;color:#f2f2f2;border-radius:6px;padding:6px 8px;cursor:pointer;text-align:left}}
button.active{{background:#f2f2f2;color:#111;border-color:#f2f2f2}}
</style>
</head>
<body>
<canvas id="c"></canvas>
<div id="hud">
  <strong>{titre}</strong><br>
  Souris: tourner | molette: zoom | double clic: reset
  <div id="controls">
    {'<label>Captures <input id="sliderN" type="range" min="1" max="8" step="1" value="'+str(n_defaut)+'"><span id="nValue">'+str(n_defaut)+'</span></label>' if avec_slider else '<span>Captures utilisees : '+str(n_defaut)+'</span>'}
    <button id="togglePlan" class="active">Plan visible</button>
    <button id="toggleGradient" class="{'active' if gradient_defaut else ''}">Gradient profondeur</button>
    <span id="count"></span>
  </div>
</div>
<script>
const allData = {json.dumps(donnees, separators=(',', ':'))};
const defaultN = "{n_defaut}";
const planThickness = {EPAISSEUR_PLAN_MM};
let currentN = defaultN;
let showPlan = true;
let gradient = {str(gradient_defaut).lower()};
let points = [];
let rx = -0.85, ry = 0.0, rz = 0.02, zoom = 0.9;
let dragging = false, lx = 0, ly = 0;
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
const slider = document.getElementById('sliderN');
const nValue = document.getElementById('nValue');
const count = document.getElementById('count');
const btnPlan = document.getElementById('togglePlan');
const btnGradient = document.getElementById('toggleGradient');
function loadPoints() {{
  points = allData[currentN].map(p => [p[0], p[1], p[2], Math.abs(p[2]) <= planThickness]);
  count.textContent = `Points affiches : ${{visiblePoints().length.toLocaleString('fr-FR')}}`;
  draw();
}}
function visiblePoints() {{
  return showPlan ? points : points.filter(p => !p[3]);
}}
function resize() {{ canvas.width = innerWidth * devicePixelRatio; canvas.height = innerHeight * devicePixelRatio; draw(); }}
function rot(p) {{
  let [x,y,z] = p;
  let cy=Math.cos(ry), sy=Math.sin(ry), cx=Math.cos(rx), sx=Math.sin(rx), cz=Math.cos(rz), sz=Math.sin(rz);
  let x1=x*cy+z*sy, z1=-x*sy+z*cy, y1=y;
  let y2=y1*cx-z1*sx, z2=y1*sx+z1*cx, x2=x1;
  return [x2*cz-y2*sz, x2*sz+y2*cz, z2];
}}
function color(t) {{
  t = Math.max(0, Math.min(1, t));
  const r = Math.round(255*Math.min(1, Math.max(0, 1.5-Math.abs(4*t-3))));
  const g = Math.round(255*Math.min(1, Math.max(0, 1.5-Math.abs(4*t-2))));
  const b = Math.round(255*Math.min(1, Math.max(0, 1.5-Math.abs(4*t-1))));
  return `rgb(${{r}},${{g}},${{b}})`;
}}
function draw() {{
  const w=canvas.width, h=canvas.height;
  ctx.fillStyle='#101114'; ctx.fillRect(0,0,w,h);
  const pts = visiblePoints();
  if (!pts.length) return;
  const xs=pts.map(p=>p[0]), ys=pts.map(p=>p[1]), zs=pts.map(p=>p[2]);
  const cx=(Math.min(...xs)+Math.max(...xs))/2, cy=(Math.min(...ys)+Math.max(...ys))/2, cz=(Math.min(...zs)+Math.max(...zs))/2;
  const scaleBase=Math.max(Math.max(...xs)-Math.min(...xs), Math.max(...ys)-Math.min(...ys), Math.max(...zs)-Math.min(...zs), 1);
  const zMin=Math.min(...zs), zMax=Math.max(...zs);
  const s=Math.min(w,h)*zoom/scaleBase;
  const projected=pts.map(p=>{{const q=rot([p[0]-cx,p[1]-cy,p[2]-cz]); return [w/2+q[0]*s,h/2-q[1]*s,q[2],p[2],p[3]];}}).sort((a,b)=>a[2]-b[2]);
  const px=Math.max(1.8, 2.7*devicePixelRatio);
  for (const p of projected) {{
    if (p[4]) ctx.fillStyle = '#d8d8d8';
    else ctx.fillStyle = gradient ? color((p[3]-zMin)/(zMax-zMin || 1)) : '#1f5fbf';
    ctx.fillRect(p[0], p[1], px, px);
  }}
}}
if (slider) slider.addEventListener('input', e => {{ currentN = e.target.value; nValue.textContent = currentN; loadPoints(); }});
btnPlan.addEventListener('click', () => {{ showPlan=!showPlan; btnPlan.classList.toggle('active', showPlan); btnPlan.textContent=showPlan?'Plan visible':'Plan masque'; loadPoints(); }});
btnGradient.addEventListener('click', () => {{ gradient=!gradient; btnGradient.classList.toggle('active', gradient); draw(); }});
canvas.addEventListener('mousedown', e => {{ dragging=true; lx=e.clientX; ly=e.clientY; }});
addEventListener('mouseup', () => dragging=false);
addEventListener('mousemove', e => {{ if(!dragging) return; ry += (e.clientX-lx)*0.006; rx += (e.clientY-ly)*0.006; lx=e.clientX; ly=e.clientY; draw(); }});
canvas.addEventListener('wheel', e => {{ e.preventDefault(); zoom *= Math.exp(-e.deltaY*0.001); draw(); }}, {{passive:false}});
canvas.addEventListener('dblclick', () => {{ rx=-0.85; ry=0; rz=0.02; zoom=0.9; draw(); }});
addEventListener('resize', resize);
btnGradient.classList.toggle('active', gradient);
loadPoints();
resize();
</script>
</body>
</html>"""


def generer_index() -> None:
    liens = "\n".join(
        f'      <a href="{url}">{titre}<span>{"Reference de la reconstruction" if nom == "plan" else "Objet reconstruit"}</span></a>'
        for nom, titre, url, _ in OBJETS
    )
    (WEB / "index.html").write_text(
        f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Reconstructions 3D - Projet PRONTO</title>
  <style>
    body {{ margin:0; min-height:100vh; font-family:Arial,sans-serif; background:#f7f7f5; color:#171717; display:grid; place-items:center; }}
    main {{ width:min(980px, calc(100vw - 40px)); padding:34px 0; }}
    h1 {{ margin:0 0 10px; font-size:32px; font-weight:700; }}
    p {{ margin:0 0 24px; color:#555; line-height:1.5; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:12px; }}
    a {{ display:block; min-height:72px; padding:18px 20px; border:1px solid #d9d9d4; border-radius:8px; background:#fff; color:#171717; text-decoration:none; font-weight:700; }}
    a span {{ display:block; margin-top:6px; color:#666; font-size:14px; font-weight:400; }}
    a:hover {{ border-color:#111; }}
  </style>
</head>
<body>
  <main>
    <h1>Reconstructions 3D</h1>
    <p>Vues interactives des reconstructions finales de notre projet PRONTO.</p>
    <div class="grid">
{liens}
    </div>
  </main>
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> None:
    mr, me = charger_matrices_calibration()
    for ancien in ["maisonette.html", "pyramide-de-keops.html", "pyramide-inclinee.html", "pyramide.html"]:
        chemin = WEB / ancien
        if chemin.exists():
            chemin.unlink()

    for nom, titre, fichier, slider in OBJETS:
        data = donnees_objet(nom, slider, mr, me)
        gradient_defaut = nom.lower() != "plan"
        (WEB / fichier).write_text(html_vue(titre, data, slider, gradient_defaut), encoding="utf-8")

    generer_index()
    (WEB / "README.md").write_text(
        "# Reconstructions 3D - Projet PRONTO\n\n"
        "Site statique contenant les vues HTML interactives des reconstructions retenues pour le rapport.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
