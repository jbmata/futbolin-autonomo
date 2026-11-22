"""
dashboard.py
------------
Panel web local para seguir un entrenamiento. Lee los logs de TensorBoard que
escribe train.py y se actualiza solo cada 10 s. No depende de nada fuera de lo
que ya trae el entorno (tensorboard viene con stable-baselines3[extra]) y solo
escucha en 127.0.0.1.

Uso:
    python dashboard.py --log-dir logs/ppo_v4 --total 4000000
    # y abrir http://127.0.0.1:8765

    # Instantánea en un solo fichero HTML (sin servidor): para guardarla, compartirla
    # o descargarla de un entrenamiento en GitHub Actions
    python dashboard.py --log-dir logs/ppo_v4 --total 4000000 --export panel.html
"""

import argparse
import glob
import json
import os
import re
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PHASE_NAMES = {
    0: "Fase 0 · rival ausente",
    1: "Fase 1 · rival flojo (0,2–0,6)",
    2: "Fase 2 · rival completo (0,4–1,0)",
    3: "Fase 3 · self-play",
}
RUNNING_WINDOW = 180.0      # s sin eventos nuevos → se considera parado
RATE_WINDOW    = 600.0      # s sobre los que se mide el ritmo
SCALARS = {
    "ep_rew":  "rollout/ep_rew_mean",
    "scored":  "game/goals_scored",
    "conc":    "game/goals_conceded",
    "shots":   "game/shots_on_target",
    "touches": "game/touches_per_ep",
    "ev_diff": "eval/goal_difference",
    "ev_pts":  "eval/points",
    "ev_win":  "eval/win_rate",
    "ev_draw": "eval/draw_rate",
    "ev_loss": "eval/loss_rate",
    "ev_for":  "eval/goals_scored",
    "ev_against": "eval/goals_conceded",
    "ev_rew":  "eval/mean_reward",
    "expvar":  "train/explained_variance",
    "std":     "train/std",
    "kl":      "train/approx_kl",
    "lr":      "train/learning_rate",
}


GEN_RE = re.compile(r"^gen\s+(\d+)\s+mejor\s+([+-]?\d+\.\d+)\s+media\s+([+-]?\d+\.\d+)"
                    r"\s+sigma\s+([\d.]+)\s+([\d.]+)\s+min")
VAL_RE = re.compile(r"^(por_defecto|optimizado|mejor_candidato)\s+([\d.]+)\s+([\d.]+)\s+([+-][\d.]+)"
                    r"\D+?([\d.]+)\s+([\d.]+)\s*$")


def read_teacher(path):
    """Lee el log de optimize_teacher.py: generaciones y tabla de validación."""
    if not path or not os.path.isfile(path):
        return None
    gens, val = [], []
    with open(path, "rb") as f:
        text = f.read().replace(b"\x00", b"").decode("utf-8", errors="replace")
    for line in text.splitlines():
        m = GEN_RE.match(line.strip())
        if m:
            g, best, mean, sigma, minutes = m.groups()
            gens.append([int(g), float(best), float(mean), float(sigma), float(minutes)])
            continue
        m = VAL_RE.match(line.strip())
        if m:
            n, sc, co, df, ci, sh = m.groups()
            val.append({"nombre": n, "marca": float(sc), "recibe": float(co),
                        "dif": float(df), "ci": float(ci), "tiros": float(sh)})
    return {"gens": gens, "validacion": val, "actualizado": os.path.getmtime(path)}


def read_plan(path):
    """Estado de las etapas del plan, escrito a mano en un JSON."""
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


class Reader:
    """Mantiene un EventAccumulator por fase y lee solo lo nuevo."""

    def __init__(self, log_dir: str, teacher_log: str = None, plan: str = None):
        self.teacher_log = teacher_log
        self.plan = plan
        self.log_dir = log_dir
        self.acc = {}
        self.lock = threading.Lock()
        self._cache = None
        self._cache_t = 0.0

    def _phase_dirs(self):
        out = []
        for d in glob.glob(os.path.join(self.log_dir, "ppo_phase*")):
            m = re.search(r"ppo_phase(\d+)", os.path.basename(d))
            if m and os.path.isdir(d):
                out.append((int(m.group(1)), d))
        return sorted(out)

    def state(self, total: int) -> dict:
        with self.lock:
            if self._cache is not None and time.time() - self._cache_t < 4.0:
                return self._cache
            self._cache = self._build(total)
            self._cache_t = time.time()
            return self._cache

    def _build(self, total: int) -> dict:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        series = {k: [] for k in SCALARS}
        timeline = []            # (wall_time, step) de cada iteración
        phases = []              # (fase, paso_inicial, paso_final)
        for phase, d in self._phase_dirs():
            if d not in self.acc:
                self.acc[d] = EventAccumulator(d, size_guidance={"scalars": 0})
            ea = self.acc[d]
            ea.Reload()
            tags = set(ea.Tags().get("scalars", []))
            steps = []
            for key, tag in SCALARS.items():
                if tag in tags:
                    pts = [(e.step, e.value) for e in ea.Scalars(tag)]
                    series[key].extend(pts)
                    if key == "ep_rew":
                        steps = [e.step for e in ea.Scalars(tag)]
                        timeline.extend((e.wall_time, e.step) for e in ea.Scalars(tag))
            if steps:
                phases.append((phase, min(steps), max(steps)))
        for k in series:
            series[k].sort()
        timeline.sort()

        now = time.time()
        step = timeline[-1][1] if timeline else 0
        last_t = timeline[-1][0] if timeline else 0.0
        running = bool(timeline) and now - last_t < RUNNING_WINDOW
        rate = None
        if len(timeline) > 3:
            t1, s1 = timeline[-1]
            old = [x for x in timeline if t1 - x[0] <= RATE_WINDOW]
            t0, s0 = old[0]
            if t1 - t0 > 30:
                rate = (s1 - s0) / (t1 - t0)
        teacher = read_teacher(self.teacher_log)
        if teacher:
            series["t_best"] = [(g[0], g[1]) for g in teacher["gens"]]
            series["t_mean"] = [(g[0], g[2]) for g in teacher["gens"]]
        started = timeline[0][0] if timeline else None
        eta = (total - step) / rate if rate and rate > 0 else None
        return {
            "now": now, "running": running, "step": step, "total": total,
            "phase": phases[-1][0] if phases else None,
            "phase_name": PHASE_NAMES.get(phases[-1][0], "") if phases else "",
            "phases": [{"phase": p, "name": PHASE_NAMES.get(p, f"Fase {p}"), "from": a, "to": b}
                       for p, a, b in phases],
            "rate": rate, "eta": eta, "started": started, "last_event": last_t or None,
            "elapsed": (last_t - started) if started else None,
            "teacher": teacher, "plan": read_plan(self.plan),
            "series": series,
        }


HTML = r"""<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Entrenamiento Futbolín</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#1c2330;--mute:#6b7585;--line:#e3e6ec;--grid:#eceff3;
--c1:#2563eb;--c2:#e5484d;--c3:#16a34a;--c4:#d97706;--c5:#7c3aed;--ok:#16a34a;--bad:#e5484d}
@media (prefers-color-scheme:dark){:root{--bg:#0f141b;--card:#171e28;--ink:#e6eaf0;--mute:#8d98a8;
--line:#26303d;--grid:#1f2833;--c1:#5b8def;--c2:#f0686d;--c3:#3fc77a;--c4:#f0a23a;--c5:#a78bfa;
--ok:#3fc77a;--bad:#f0686d}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,Segoe UI,Roboto,sans-serif}
main{max-width:1180px;margin:0 auto;padding:20px 16px 40px}
h1{font-size:20px;margin:0 0 2px}
.sub{color:var(--mute);margin-bottom:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.top{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-bottom:12px}
.k{color:var(--mute);font-size:12px;text-transform:uppercase;letter-spacing:.04em}
.v{font-size:22px;font-weight:650;margin-top:2px;font-variant-numeric:tabular-nums}
.v small{font-size:13px;font-weight:500;color:var(--mute)}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px;background:var(--bad)}
.dot.on{background:var(--ok);box-shadow:0 0 0 3px color-mix(in srgb,var(--ok) 25%,transparent)}
.bar{height:10px;background:var(--grid);border-radius:6px;overflow:hidden;margin-top:8px;position:relative}
.bar>i{display:block;height:100%;background:var(--c1)}
.bar>b{position:absolute;top:0;bottom:0;width:2px;background:var(--card);opacity:.9}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:12px;margin-top:12px}
.card h2{font-size:14px;margin:0 0 2px}
.card p{margin:0 0 8px;color:var(--mute);font-size:12px}
canvas{width:100%;height:210px;display:block}
.legend{display:flex;gap:14px;flex-wrap:wrap;margin-top:6px;font-size:12px;color:var(--mute)}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:-1px}
#tip{position:fixed;pointer-events:none;background:var(--ink);color:var(--bg);padding:6px 9px;border-radius:7px;
font-size:12px;display:none;white-space:nowrap;z-index:9}
.note{color:var(--mute);font-size:12px;margin-top:14px}
.err{color:var(--bad)}
.et{display:flex;gap:10px;align-items:baseline;padding:5px 0;border-top:1px solid var(--line)}
.et:first-child{border-top:0}
.et b{min-width:92px;font-size:12px;text-transform:uppercase;letter-spacing:.03em}
.et .d{color:var(--mute);font-size:12px}
.s-hecho b{color:var(--ok)}.s-curso b{color:var(--c4)}.s-pendiente b{color:var(--mute)}.s-error b{color:var(--bad)}
table.v{border-collapse:collapse;margin-top:6px;font-variant-numeric:tabular-nums}
table.v td,table.v th{padding:3px 12px 3px 0;text-align:right;font-size:12px}
table.v th:first-child,table.v td:first-child{text-align:left}
</style></head><body><main>
<h1>Entrenamiento del futbolín</h1>
<div class="sub" id="sub">Cargando…</div>
<div class="card" id="plan" style="display:none;margin-bottom:12px"><div class="k" id="plan_t">Plan</div><div id="plan_l"></div></div>
<div class="top">
  <div class="card"><div class="k">Estado</div><div class="v" id="state">—</div><div class="sub" id="phase" style="margin:2px 0 0"></div></div>
  <div class="card"><div class="k">Pasos</div><div class="v" id="steps">—</div><div class="bar"><i id="pbar" style="width:0"></i></div></div>
  <div class="card"><div class="k">Ritmo</div><div class="v" id="rate">—</div></div>
  <div class="card"><div class="k">Tiempo restante</div><div class="v" id="eta">—</div><div class="sub" id="etaat" style="margin:2px 0 0"></div></div>
  <div class="card"><div class="k">Transcurrido</div><div class="v" id="elapsed">—</div></div>
</div>
<div class="grid" id="charts"></div>
<div class="note" id="note"></div>
</main><div id="tip"></div>
<script>
const CH=[
 {id:"teacher",gen:true,t:"Profesor: optimización de parámetros (CMA-ES)",p:"Goles de diferencia por minuto (más una parte por tiros) por generación. Mucho ruido: cada candidato juega pocos partidos.",
  s:[["t_best","Mejor candidato","--c3"],["t_mean","Media de la población","--c1"]],zero:true},
 {id:"goals",t:"Goles por partida (entrenamiento)",p:"Media de las últimas 100 partidas. Si lo marcado supera a lo recibido, va ganando.",
  s:[["scored","Marcados","--c1"],["conc","Recibidos","--c2"]],min:0},
 {id:"evd",t:"Evaluación: diferencia de goles",p:"Contra el heurístico de nivel 0,7, siempre las mismas partidas. Por encima de 0 es mejor que él.",
  s:[["ev_diff","Diferencia","--c1"]],zero:true},
 {id:"evw",t:"Evaluación: resultados",p:"Proporción de partidas ganadas, empatadas y perdidas.",
  s:[["ev_win","Ganadas","--c3"],["ev_draw","Empatadas","--c4"],["ev_loss","Perdidas","--c2"]],min:0,max:1},
 {id:"shots",t:"Tiros a puerta por partida",p:"Veces que un toque propio deja la bola dirigida a la portería rival.",
  s:[["shots","Tiros","--c5"]],min:0},
 {id:"rew",t:"Recompensa media por partida",p:"Incluye goles y las ayudas de potencial; sube si juega mejor.",
  s:[["ep_rew","Entrenamiento","--c1"],["ev_rew","Evaluación","--c4"]],zero:true},
 {id:"health",t:"Salud del entrenamiento",p:"Explained variance cerca de 1 = el crítico predice bien. Si std se desploma, deja de explorar.",
  s:[["expvar","Explained variance","--c3"],["std","std de las acciones","--c5"]],min:0}
];
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const hasPPO=()=>DATA&&DATA.series.ep_rew.length>0;
const box=document.getElementById("charts");
CH.forEach(c=>{const d=document.createElement("div");d.className="card";
 d.innerHTML=`<h2>${c.t}</h2><p>${c.p}</p><canvas id="cv_${c.id}"></canvas><div class="legend">`+
 c.s.map(s=>`<span><i style="background:var(${s[2]})"></i>${s[1]}</span>`).join("")+`</div>`;box.appendChild(d)});
const fmt={n:(x,d=0)=>x==null?"—":x.toLocaleString("es-ES",{maximumFractionDigits:d}),
 dur:s=>{if(s==null)return"—";s=Math.max(0,Math.round(s));const h=Math.floor(s/3600),m=Math.floor(s%3600/60);
  return h?`${h} h ${String(m).padStart(2,"0")} min`:`${m} min`}};
let DATA=null,HOVER=null;
function nice(lo,hi){if(lo===hi){lo-=1;hi+=1}const r=hi-lo,p=Math.pow(10,Math.floor(Math.log10(r))),
 st=[1,2,2.5,5,10].map(m=>m*p).find(s=>r/s<=5)||10*p;return[Math.floor(lo/st)*st,Math.ceil(hi/st)*st,st]}
function draw(c){
 const cv=document.getElementById("cv_"+c.id),dpr=devicePixelRatio||1,W=cv.clientWidth,H=cv.clientHeight;
 cv.width=W*dpr;cv.height=H*dpr;const g=cv.getContext("2d");g.scale(dpr,dpr);
 const L=44,R=10,T=8,B=22;
 const total=c.gen?Math.max(1,...DATA.series.t_best.map(p=>p[0]),16):DATA.total;
 let lo=c.min??Infinity,hi=c.max??-Infinity;
 c.s.forEach(([k])=>DATA.series[k].forEach(([,v])=>{if(c.min==null)lo=Math.min(lo,v);if(c.max==null)hi=Math.max(hi,v)}));
 if(c.zero){lo=Math.min(lo,0);hi=Math.max(hi,0)}
 if(!isFinite(lo)){lo=0}if(!isFinite(hi)){hi=1}
 if(c.max==null||c.min==null){const n=nice(lo,hi);if(c.min==null)lo=n[0];if(c.max==null)hi=n[1];if(hi<=lo)hi=lo+1}
 const X=s=>L+(W-L-R)*s/total,Y=v=>T+(H-T-B)*(1-(v-lo)/(hi-lo));
 g.font="11px system-ui";g.textBaseline="middle";g.fillStyle=css("--mute");g.strokeStyle=css("--grid");g.lineWidth=1;
 const n=nice(lo,hi);for(let v=Math.ceil(lo/n[2])*n[2];v<=hi+1e-9;v+=n[2]){const y=Y(v);
  g.beginPath();g.moveTo(L,y);g.lineTo(W-R,y);g.stroke();g.textAlign="right";g.fillText(+v.toFixed(3),L-6,y)}
 g.textAlign="center";[0,.25,.5,.75,1].forEach(f=>g.fillText(c.gen?String(Math.round(total*f)):fmt.n(total*f/1e6,2)+"M",X(total*f),H-9));
 (c.gen?[]:DATA.phases).forEach((p,i)=>{if(i){g.strokeStyle=css("--line");g.setLineDash([3,3]);g.beginPath();
  g.moveTo(X(p.from),T);g.lineTo(X(p.from),H-B);g.stroke();g.setLineDash([])}
  g.fillStyle=css("--mute");g.textAlign="left";g.fillText("F"+p.phase,X(p.from)+4,T+7)});
 if(c.zero||lo<0&&hi>0){g.strokeStyle=css("--mute");g.globalAlpha=.5;g.beginPath();g.moveTo(L,Y(0));g.lineTo(W-R,Y(0));g.stroke();g.globalAlpha=1}
 c.s.forEach(([k,,col])=>{const pts=DATA.series[k];if(!pts.length)return;
  g.strokeStyle=css(col);g.lineWidth=2;g.lineJoin="round";g.beginPath();
  pts.forEach(([s,v],i)=>{i?g.lineTo(X(s),Y(v)):g.moveTo(X(s),Y(v))});g.stroke();
  if(pts.length<40){g.fillStyle=css(col);pts.forEach(([s,v])=>{g.beginPath();g.arc(X(s),Y(v),2.5,0,7);g.fill()})}});
 if(HOVER&&HOVER.id===c.id){g.strokeStyle=css("--mute");g.beginPath();g.moveTo(HOVER.x,T);g.lineTo(HOVER.x,H-B);g.stroke()}
 cv._m={L,R,W,total,X}
}
function render(){if(!DATA)return;CH.forEach(draw)}
function tip(e,c){const cv=e.target,m=cv._m,r=cv.getBoundingClientRect(),x=e.clientX-r.left;
 const step=(x-m.L)/(m.W-m.L-m.R)*m.total;HOVER={id:c.id,x};
 const rows=c.s.map(([k,n,col])=>{const pts=DATA.series[k];if(!pts.length)return"";
  let b=pts[0];for(const p of pts)if(Math.abs(p[0]-step)<Math.abs(b[0]-step))b=p;
  return `<div><span style="color:var(${col})">●</span> ${n}: <b>${fmt.n(b[1],3)}</b></div>`}).join("");
 const t=document.getElementById("tip");t.innerHTML=`<div>${c.gen?"generación "+Math.round(step):fmt.n(step/1e3,0)+"k pasos"}</div>`+rows;
 t.style.display="block";t.style.left=Math.min(e.clientX+14,innerWidth-190)+"px";t.style.top=e.clientY+14+"px";render()}
CH.forEach(c=>{const cv=document.getElementById("cv_"+c.id);cv.onmousemove=e=>tip(e,c);
 cv.onmouseleave=()=>{HOVER=null;document.getElementById("tip").style.display="none";render()}});
addEventListener("resize",render);
const last=k=>{const p=DATA.series[k];return p.length?p[p.length-1][1]:null};
async function tick(){
 try{DATA=window.__STATE__||await (await fetch("/api/state")).json();
  const d=DATA,pct=d.step/d.total*100;
  document.getElementById("sub").textContent=window.__STATE__?"Instantánea tomada el "+new Date(d.now*1000).toLocaleString("es-ES"):"Actualizado "+new Date(d.now*1000).toLocaleTimeString("es-ES")+" · se refresca cada 10 s";
  document.getElementById("state").innerHTML=`<span class="dot ${d.running?"on":""}"></span>${d.running?"Entrenando":(d.step?"Parado":"Sin datos")}`;
  document.getElementById("phase").textContent=d.phase_name;
  document.getElementById("steps").innerHTML=`${fmt.n(d.step/1e6,2)} M <small>de ${fmt.n(d.total/1e6,1)} M · ${fmt.n(pct,1)} %</small>`;
  document.getElementById("pbar").style.width=pct+"%";
  document.getElementById("rate").innerHTML=d.rate?`${fmt.n(d.rate)} <small>pasos/s</small>`:"—";
  document.getElementById("eta").textContent=d.running?fmt.dur(d.eta):"—";
  document.getElementById("etaat").textContent=d.running&&d.eta?"termina hacia las "+new Date((d.now+d.eta)*1000).toLocaleTimeString("es-ES",{hour:"2-digit",minute:"2-digit"}):"";
  document.getElementById("elapsed").textContent=fmt.dur(d.elapsed);
  const ed=last("ev_diff"),sc=last("scored"),co=last("conc");
  document.getElementById("note").innerHTML=
   (sc!=null?`Entrenamiento: marca <b>${fmt.n(sc,2)}</b> y recibe <b>${fmt.n(co,2)}</b> goles por partida. `:"")+
   (ed!=null?`Última evaluación: diferencia <b>${fmt.n(ed,2)}</b>, ganadas ${fmt.n(last("ev_win")*100)} %, empatadas ${fmt.n(last("ev_draw")*100)} %, perdidas ${fmt.n(last("ev_loss")*100)} %. `:"")+
   `<br>Las curvas de evaluación usan pocas partidas y oscilan mucho; fíate de la tendencia de varias evaluaciones, no de un punto. `+
   `La estimación de tiempo se calcula con el ritmo de los últimos 10 min y suele subir en las fases con rival más complejo.`;
  const pl=d.plan,pe=document.getElementById("plan");
  if(pl&&pl.etapas){pe.style.display="block";document.getElementById("plan_t").textContent=pl.titulo||"Plan";
   const nm={hecho:"Hecho",curso:"En curso",pendiente:"Pendiente",error:"Problema"};
   document.getElementById("plan_l").innerHTML=pl.etapas.map(e=>`<div class="et s-${e.estado}"><b>${nm[e.estado]||e.estado}</b><span>${e.nombre}<br><span class="d">${e.detalle||""}</span></span></div>`).join("")}
  const vt=d.teacher&&d.teacher.validacion&&d.teacher.validacion.length?
   `<br><b>Validación del profesor</b> (partidos nuevos, goles por minuto):<table class="v"><tr><th></th><th>marca</th><th>recibe</th><th>diferencia</th><th>± IC 95 %</th><th>tiros</th></tr>`+
   d.teacher.validacion.map(v=>`<tr><td>${v.nombre.replace("_"," ")}</td><td>${v.marca.toFixed(2)}</td><td>${v.recibe.toFixed(2)}</td><td>${(v.dif>0?"+":"")+v.dif.toFixed(2)}</td><td>${v.ci.toFixed(2)}</td><td>${v.tiros.toFixed(1)}</td></tr>`).join("")+`</table>`:"";
  if(vt)document.getElementById("note").insertAdjacentHTML("beforeend",vt);
  render()}catch(e){document.getElementById("sub").innerHTML='<span class="err">Sin conexión con el panel</span>'}
}
tick();setInterval(tick,10000);
</script></body></html>"""


def export_html(log_dir: str, total: int, path: str, teacher_log: str = None, plan: str = None):
    """Escribe el panel con los datos incrustados: un único fichero que no necesita servidor."""
    state = Reader(log_dir, teacher_log, plan).state(total)
    data = json.dumps(state).replace("</", "<\\/")      # evita cerrar el <script> por error
    marker = "<script>\nconst CH="
    assert HTML.count(marker) == 1
    html = HTML.replace(marker, "<script>window.__STATE__=" + data + ";</script>\n" + marker, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return state


def make_handler(reader: Reader, total: int):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, body, ctype):
            data = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.split("?")[0] == "/api/state":
                try:
                    self._send(200, json.dumps(reader.state(total)), "application/json")
                except Exception as e:      # el panel no debe caerse por un log a medio escribir
                    self._send(500, json.dumps({"error": str(e)}), "application/json")
            elif self.path.split("?")[0] in ("/", "/index.html"):
                self._send(200, HTML, "text/html; charset=utf-8")
            else:
                self._send(404, "no encontrado", "text/plain; charset=utf-8")
    return Handler


def main():
    ap = argparse.ArgumentParser(description="Panel de seguimiento del entrenamiento")
    ap.add_argument("--log-dir", default="logs/ppo_v4")
    ap.add_argument("--total", type=int, default=4_000_000, help="Pasos totales del entrenamiento")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-open", action="store_true", help="No abrir el navegador")
    ap.add_argument("--teacher-log", default=None,
                    help="Log de optimize_teacher.py para dibujar la optimización del profesor")
    ap.add_argument("--plan", default=None,
                    help="JSON con las etapas del plan (titulo y etapas: nombre, estado, detalle)")
    ap.add_argument("--wait", action="store_true",
                    help="Si el directorio de logs aún no existe, crearlo y esperar a que el entrenamiento escriba")
    ap.add_argument("--export", metavar="FICHERO.html",
                    help="Escribir una instantánea del panel en un fichero y salir")
    args = ap.parse_args()
    if args.wait:
        os.makedirs(args.log_dir, exist_ok=True)
    if not os.path.isdir(args.log_dir):
        raise SystemExit(f"No existe {args.log_dir}")
    if args.export:
        st = export_html(args.log_dir, args.total, args.export, args.teacher_log, args.plan)
        print(f"Instantánea guardada en {args.export} (paso {st['step']:,} de {args.total:,})")
        return
    server = ThreadingHTTPServer(("127.0.0.1", args.port),
                                make_handler(Reader(args.log_dir, args.teacher_log, args.plan), args.total))
    url = f"http://127.0.0.1:{args.port}"
    print(f"Panel en {url}  (Ctrl+C para cerrar)")
    if not args.no_open:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
