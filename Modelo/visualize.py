"""
visualize.py
------------
Graba un partido del simulador y lo convierte en un reproductor en un solo fichero
HTML (sin servidor ni dependencias): pausa, cámara lenta, avance por fotogramas y
saltos a goles, saques y tiros.

Dos vistas a la misma escala en X:
  - Planta: bola real, bola que ve la cámara y figuras.
  - Alzado: cada figura con su cabeza, pierna y taco girados, y la bola. Verde si la
    bola cabe por debajo de la figura, del color del equipo si la tapa. El contorno
    discontinuo es donde el firmware CREE que está la barra (pasos perdidos).

Uso:
    python visualize.py --teacher                              # profesor optimizado
    python visualize.py --teacher models/teacher/roles.json
    python visualize.py --heuristic                            # heurístico por defecto
    python visualize.py --model models/dagger/student          # una red entrenada
    python visualize.py --teacher --seconds 60 --skill 0.7 --seed 3 --save partido.html
"""

import argparse
import json
import os
import webbrowser

import numpy as np

import world
from env import FoosballEnv, RoleParams
from example import HeuristicAgent
from world import BAR_DEFINITIONS

DEFAULT_TEACHER = "models/teacher/heuristic_optimized.json"


def make_agent(env, model=None, teacher=None):
    """Devuelve una función obs -> acción."""
    if model is not None:
        return lambda obs: model.predict(obs, deterministic=True)[0]
    if teacher is not None:
        agent = HeuristicAgent(env, seed=0, reaction_delay=0.0, params=RoleParams.load(teacher))
    else:
        agent = HeuristicAgent(env, seed=0)
    return lambda obs: agent.predict()


def record(seconds=30.0, seed=0, skill=0.7, opponent="heuristic", model_path=None, teacher=None,
           title="") -> dict:
    """Juega un partido y devuelve todo lo necesario para reproducirlo."""
    model, version = None, None
    if model_path:
        from stable_baselines3 import PPO
        from train import obs_version_of
        model = PPO.load(model_path, device="cpu")
        version = obs_version_of(model)
    env = FoosballEnv(max_steps=int(round(seconds / 0.02)), opponent=opponent, opponent_skill=skill,
                      obs_version=version)
    act = make_agent(env, model, teacher)
    obs, _ = env.reset(seed=seed)
    cam = env._builder.camera if env._builder is not None else None
    frames, events = [], []

    def snap():
        b = env.state.ball
        row = [b.x, b.y, b.vx, b.vy]
        row += [cam.ball[0], cam.ball[1], float(cam.visible)] if cam else [b.x, b.y, 1.0]
        for bar in env.state.bars:
            row += [bar.linear_pos, bar.angle]
        for i in range(8):
            lin, ang, _, _ = env.actuators[i].estimate()
            row += [lin, ang]
        frames.append([round(float(v), 4) for v in row])

    snap()
    done = False
    pinned = False
    while not done:
        obs, _, term, trunc, info = env.step(act(obs))
        done = term or trunc
        snap()
        ev, k = info["events"], len(frames) - 1
        if ev["goal"]:
            events.append([k, "gol", f"Gol de {ev['goal']} ({info['score']['A']}-{info['score']['B']})"])
        if ev.get("serve"):
            events.append([k, "saque", f"Saca el portero de {ev['serve']}"])
        if ev["shot_on_target"]:
            if not events or events[-1][1] != "tiro" or k - events[-1][0] > 10:
                events.append([k, "tiro", f"Tiro a puerta de {ev['shot_on_target']}"])
        if ev["pinned"] and not pinned:
            events.append([k, "pisada", "Bola pisada"])
        pinned = ev["pinned"]
    return {
        "title": title, "dt": env.dt, "frames": frames, "events": events,
        "score": info["score"], "lost_steps": info.get("lost_steps", 0),
        "bars": [{"x": x, "team": t, "role": r, "off": list(env.state.bars[i].player_offsets)}
                 for i, (x, n, t, r) in enumerate(BAR_DEFINITIONS)],
        "geo": {"L": world.FIELD_LENGTH, "W": world.FIELD_WIDTH, "goal": world.GOAL_WIDTH,
                "r": world.BALL_RADIUS, "H": world.ROD_HEIGHT, "foot": world.FOOT_LENGTH,
                "head": world.HEAD_LENGTH, "fh": world.FOOT_HEIGHT, "fl": world.FOOT_HALF_LENGTH,
                "body": world.BODY_HALF_DEPTH, "leg": world.PLAYER_DEPTH, "hw": world.PLAYER_HALF_WIDTH},
    }


HTML = r"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Partido simulado</title><style>
:root{--bg:#10151c;--card:#181f29;--ink:#e8ecf2;--mute:#93a0b2;--line:#2a3442;--a:#5b9bff;--b:#ff6b6b;--ok:#4fd08a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,Segoe UI,sans-serif}
main{max-width:1240px;margin:0 auto;padding:16px}h1{font-size:17px;margin:0}
.row{display:flex;gap:14px;align-items:center;flex-wrap:wrap;justify-content:space-between;margin-bottom:10px}
.sc{font-size:26px;font-weight:700;font-variant-numeric:tabular-nums}.sc .a{color:var(--a)}.sc .b{color:var(--b)}
.mute{color:var(--mute);font-size:12px}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px;margin-bottom:10px}
canvas{width:100%;display:block;border-radius:6px}button,select{background:#232d3a;color:var(--ink);border:1px solid var(--line);
border-radius:7px;padding:6px 11px;font:inherit;cursor:pointer}button:hover{background:#2d3949}button:focus-visible,select:focus-visible{outline:2px solid var(--a)}
input[type=range]{width:100%;accent-color:var(--a);margin:8px 0 0}.ctl{display:flex;gap:7px;flex-wrap:wrap;align-items:center}
.ev{display:flex;gap:6px;flex-wrap:wrap;max-height:120px;overflow:auto}.ev button{padding:3px 9px;font-size:12px}
.gol{border-color:#f5c542;color:#f5c542}.saque{border-color:var(--ok);color:var(--ok)}
.key{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--mute);margin-top:6px}.key i{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:5px;vertical-align:-1px}
</style></head><body><main>
<div class="row"><div><h1 id="ti"></h1><div class="mute" id="inf"></div></div><div class="sc"><span class="a">A <span id="sa">0</span></span> – <span class="b"><span id="sb">0</span> B</span></div></div>
<div class="card"><canvas id="top"></canvas><div style="height:8px"></div><canvas id="side"></canvas>
<div class="key"><span><i style="background:#fff"></i>bola real</span><span><i style="border:2px solid #ffd54a"></i>bola según la cámara</span>
<span><i style="background:var(--ok)"></i>figura levantada: la bola pasa</span><span><i style="background:var(--a)"></i>robot (A, ataca →) tapa</span><span><i style="background:var(--b)"></i>rival (B, ataca ←) tapa</span><span>línea discontinua: dónde cree el firmware que está</span></div>
<input id="sl" type="range" min="0" value="0" aria-label="Posición en el partido">
<div class="ctl" style="margin-top:6px"><button id="pl">Pausa</button><button id="bk" title="Fotograma anterior">◀</button><button id="fw" title="Fotograma siguiente">▶</button>
<select id="sp" aria-label="Velocidad"><option value="0.1">0,1×</option><option value="0.25">0,25×</option><option value="0.5">0,5×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="4">4×</option></select>
<span class="mute" id="st"></span></div></div>
<div class="card"><div class="mute" style="margin-bottom:6px">Saltar a (empieza 1 s antes) · espacio: pausa · flechas: fotograma</div><div class="ev" id="ev"></div></div>
</main><script>(()=>{
const D=__DATA__,G=D.geo,F=D.frames,$=i=>document.getElementById(i);
const C={a:"#5b9bff",b:"#ff6b6b",ok:"#4fd08a"};let k=0,play=true,acc=0,last=0;
$("ti").textContent=D.title;$("sl").max=F.length-1;
function parts(){const L=G.foot;return[[-G.head,0,G.body],[0,L-G.fh,G.leg],[L-G.fh,L,G.fl]]}
function low(t){const s=Math.sin(t),c=Math.cos(t);let m=9;for(const[u0,u1,h]of parts())for(const u of[u0,u1])for(const v of[-h,h])m=Math.min(m,G.H-u*c+v*s);return m}
const passes=t=>low(t)>2*G.r;
function fit(cv,h){const d=devicePixelRatio||1,w=cv.clientWidth;cv.height=Math.round(w*h*d);cv.width=Math.round(w*d);cv.style.height=w*h+"px";const g=cv.getContext("2d");g.setTransform(d,0,0,d,0,0);return[g,w,w*h]}
const M=.05,SX=G.L+2*M;
function top(){const cv=$("top"),[g,w,h]=fit(cv,(G.W+.04)/SX),s=w/SX,X=x=>(x+G.L/2+M)*s,Y=y=>h/2-y*s,f=F[k];
 g.fillStyle="#0c1016";g.fillRect(0,0,w,h);g.fillStyle="#1f6b3a";g.fillRect(X(-G.L/2),Y(G.W/2),G.L*s,G.W*s);
 g.strokeStyle="#ffffff55";g.lineWidth=1;g.beginPath();g.moveTo(X(0),Y(G.W/2));g.lineTo(X(0),Y(-G.W/2));g.stroke();
 g.fillStyle="#ffffff2e";g.fillRect(X(-G.L/2-M),Y(G.goal/2),M*s,G.goal*s);g.fillRect(X(G.L/2),Y(G.goal/2),M*s,G.goal*s);
 D.bars.forEach((b,i)=>{const lin=f[7+2*i],t=f[8+2*i],base=b.team=="A"?C.a:C.b,sn=Math.sin(t),cs=Math.cos(t);
  g.strokeStyle="#c9d1db99";g.lineWidth=2;g.beginPath();g.moveTo(X(b.x),Y(G.W/2));g.lineTo(X(b.x),Y(-G.W/2));g.stroke();
  const ext=p=>{let a=9,z=-9;const[u0,u1,hh]=p;for(const u of[u0,u1])for(const v of[-hh,hh]){const x=b.x+u*sn+v*cs;a=Math.min(a,x);z=Math.max(z,x)}return[a,z]};
  b.off.forEach(o=>{const y=lin+o;[0,1,2].forEach(j=>{const[a,z]=ext(parts()[j]);g.globalAlpha=j==2?.95:.6;g.fillStyle=passes(t)?C.ok:base;g.fillRect(X(a),Y(y+G.hw),(z-a)*s,2*G.hw*s)});g.globalAlpha=1})});
 g.strokeStyle="#ffffff66";g.lineWidth=1.5;g.beginPath();for(let i=Math.max(0,k-25);i<=k;i++){const p=F[i],q=F[i-1];if(i==Math.max(0,k-25)||Math.hypot(p[0]-q[0],p[1]-q[1])>.15)g.moveTo(X(p[0]),Y(p[1]));else g.lineTo(X(p[0]),Y(p[1]))}g.stroke();
 g.fillStyle="#fff";g.beginPath();g.arc(X(f[0]),Y(f[1]),G.r*s,0,7);g.fill();
 g.strokeStyle=f[6]?"#ffd54a":"#ff9f43";g.lineWidth=2;g.setLineDash(f[6]?[]:[3,3]);g.beginPath();g.arc(X(f[4]),Y(f[5]),G.r*s,0,7);g.stroke();g.setLineDash([]);
 g.font="11px system-ui";g.textAlign="center";D.bars.forEach((b,i)=>{g.fillStyle=b.team=="A"?"#a9caff":"#ffb3b3";g.fillText(b.role+" "+b.team,X(b.x),12)})}
function side(){const cv=$("side"),Z=.15,[g,w,h]=fit(cv,Z/SX),s=w/SX,X=x=>(x+G.L/2+M)*s,Y=z=>h-10-z*s,f=F[k];
 g.fillStyle="#0c1016";g.fillRect(0,0,w,h);g.fillStyle="#1f6b3a";g.fillRect(X(-G.L/2),Y(0),G.L*s,10);
 g.strokeStyle="#ffffff30";g.setLineDash([2,4]);g.beginPath();g.moveTo(X(-G.L/2),Y(2*G.r));g.lineTo(X(G.L/2),Y(2*G.r));g.stroke();g.setLineDash([]);
 const poly=(bx,t)=>{const sn=Math.sin(t),cs=Math.cos(t);return parts().map(([u0,u1,v])=>[[u0,-v],[u1,-v],[u1,v],[u0,v]].map(([u,q])=>[X(bx+u*sn+q*cs),Y(G.H-u*cs+q*sn)]))};
 const path=p=>{g.beginPath();p.forEach(([x,y],i)=>i?g.lineTo(x,y):g.moveTo(x,y));g.closePath()};
 D.bars.forEach((b,i)=>{const t=f[8+2*i],e=f[24+2*i],col=passes(t)?C.ok:(b.team=="A"?C.a:C.b);
  poly(b.x,t).forEach(p=>{g.fillStyle=col;g.globalAlpha=.9;path(p);g.fill();g.globalAlpha=1});
  if(Math.abs(e-t)>.05){g.strokeStyle="#fff";g.setLineDash([3,3]);g.lineWidth=1;poly(b.x,e).forEach(p=>{path(p);g.stroke()});g.setLineDash([])}
  g.fillStyle="#dfe5ec";g.beginPath();g.arc(X(b.x),Y(G.H),3,0,7);g.fill();
  g.font="11px system-ui";g.textAlign="center";g.fillStyle=col;g.fillText(Math.round(t*180/Math.PI*(b.team=="A"?1:-1))+"°",X(b.x),12)});
 g.fillStyle="#fff";g.beginPath();g.arc(X(f[0]),Y(G.r),G.r*s,0,7);g.fill()}
function score(){let a=0,b=0;for(const e of D.events)if(e[0]<=k&&e[1]=="gol")e[2].startsWith("Gol de A")?a++:b++;$("sa").textContent=a;$("sb").textContent=b}
function draw(){top();side();score();$("sl").value=k;const f=F[k],n=(x,d=2)=>x.toLocaleString("es-ES",{minimumFractionDigits:d,maximumFractionDigits:d});
 const err=Math.hypot(f[4]-f[0],f[5]-f[1])*1000;$("st").textContent=`${n(k*D.dt)} s de ${n((F.length-1)*D.dt,0)} s · bola a ${n(Math.hypot(f[2],f[3]))} m/s · la cámara falla ${n(err,0)} mm${f[6]?"":" (bola oculta)"}`}
function loop(t){if(last&&play){acc+=(t-last)/1000*+$("sp").value;while(acc>=D.dt){acc-=D.dt;if(k<F.length-1)k++;else setPlay(false)}draw()}last=t;requestAnimationFrame(loop)}
function setPlay(v){play=v;acc=0;$("pl").textContent=v?"Pausa":"Reproducir"}
const go=n=>{k=Math.max(0,Math.min(F.length-1,n));draw()};
$("pl").onclick=()=>{if(!play&&k==F.length-1)k=0;setPlay(!play)};$("bk").onclick=()=>{setPlay(false);go(k-1)};$("fw").onclick=()=>{setPlay(false);go(k+1)};$("sl").oninput=e=>{setPlay(false);go(+e.target.value)};
onkeydown=e=>{if(["SELECT","BUTTON"].includes(e.target.tagName)&&e.key==" ")return;if(e.key==" "){e.preventDefault();$("pl").click()}if(e.key=="ArrowLeft"){e.preventDefault();$("bk").click()}if(e.key=="ArrowRight"){e.preventDefault();$("fw").click()}};
D.events.forEach(e=>{const b=document.createElement("button");b.className=e[1];b.textContent=(e[0]*D.dt).toFixed(1).replace(".",",")+" s · "+e[2];b.onclick=()=>{go(e[0]-Math.round(1/D.dt));setPlay(true)};$("ev").append(b)});
if(!D.events.length)$("ev").textContent="Sin eventos";
$("inf").textContent=`Ángulos en el sentido de ataque de cada equipo: negativo = pie atrás. Pasos perdidos por el robot en el partido: ${D.lost_steps}.`;
const q=new URLSearchParams(location.search);if(q.has("f")){go(+q.get("f"));setPlay(false)}
addEventListener("resize",draw);draw();requestAnimationFrame(loop);
})()</script></body></html>"""


def export(data: dict, path: str):
    text = HTML.replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"), 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main():
    ap = argparse.ArgumentParser(description="Graba un partido y lo reproduce en el navegador")
    who = ap.add_mutually_exclusive_group()
    who.add_argument("--model", help="Red entrenada (PPO o alumno DAgger), sin .zip")
    who.add_argument("--teacher", nargs="?", const=DEFAULT_TEACHER, metavar="FICHERO",
                     help="Heurístico optimizado (por defecto models/teacher/heuristic_optimized.json)")
    who.add_argument("--heuristic", action="store_true", help="Heurístico por defecto")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skill", type=float, default=0.7, help="Nivel del rival (0..1)")
    ap.add_argument("--opponent", default="heuristic", choices=("heuristic", "absent"))
    ap.add_argument("--save", default="partido.html", help="Fichero HTML de salida")
    ap.add_argument("--no-open", action="store_true", help="No abrir el navegador")
    args = ap.parse_args()

    name = args.model or args.teacher or "heurístico por defecto"
    title = f"{name} contra rival {'ausente' if args.opponent == 'absent' else f'de nivel {args.skill}'} (semilla {args.seed})"
    data = record(args.seconds, args.seed, args.skill, args.opponent, args.model, args.teacher, title)
    export(data, args.save)
    print(f"Marcador A {data['score']['A']} - {data['score']['B']} B | {len(data['events'])} eventos | {args.save}")
    if not args.no_open:
        webbrowser.open("file:///" + os.path.abspath(args.save).replace("\\", "/"))


if __name__ == "__main__":
    main()
