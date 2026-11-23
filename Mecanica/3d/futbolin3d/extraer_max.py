"""Extrae las mallas de un .max de 3ds Max (sin tener 3ds Max) con su jerarquia.

Lee el contenedor OLE (olefile), recorre los chunks de 'Scene' y reconstruye,
para cada nodo con 'Editable Mesh', su matriz de mundo (posicion, rotacion Euler
XYZ, escala, desfase del objeto y padres). Guarda un .pkl con una lista de
partes: {name, group (nombre del padre), V (vertices en mundo, cm), F (caras)}.

Uso:  python -I extraer_max.py Foosball.max... partes.pkl
Requiere: pip install olefile numpy
Limitaciones: solo controladores Position/Rotation/Scale con Euler XYZ estaticos
(los nodos con TCB Rotation / Look At, es decir luces y camaras, se ignoran).
"""
import pickle
import struct
import sys

import numpy as np
import olefile


def chunks(b, off, end, depth=0):
    while off + 6 <= end:
        tid, ln = struct.unpack_from('<HI', b, off)
        hdr = 6
        if ln == 0:
            ln = struct.unpack_from('<Q', b, off + 6)[0]
            hdr = 14
            cont = ln & (1 << 63)
            ln &= (1 << 63) - 1
        else:
            cont = ln & 0x80000000
            ln &= 0x7FFFFFFF
        if ln < hdr or off + ln > end:
            return
        yield depth, tid, off + hdr, ln - hdr, bool(cont), off
        if cont:
            yield from chunks(b, off + hdr, off + ln, depth + 1)
        off += ln


def extraer(ruta_max):
    o = olefile.OleFileIO(ruta_max)
    s = o.openstream('Scene').read()
    d = o.openstream('ClassDirectory3').read()
    names = [d[po:po + l].decode('utf-16le').rstrip('\0')
             for _, t, po, l, c, off in chunks(d, 0, len(d)) if t == 0x2042]
    objs = [(t, po, l) for dd, t, po, l, c, off in chunks(s, 0, len(s)) if dd == 1]

    def cls(i):
        return names[objs[i][0]]

    cache_k = {}

    def kids(idx):
        if idx not in cache_k:
            t, po, l = objs[idx]
            out = {}
            for dd, tt, p2, l2, c2, o2 in chunks(s, po, po + l, 0):
                if dd == 0 or tt in (0x914, 0x912):
                    out.setdefault(tt, []).append((p2, l2))
            cache_k[idx] = out
        return cache_k[idx]

    def u32s(p, l):
        return list(struct.unpack_from('<%dI' % (l // 4), s, p))

    def f32s(p, l):
        return list(struct.unpack_from('<%df' % (l // 4), s, p))

    def refs(idx):
        k = kids(idx)
        return u32s(*k[0x2034][0]) if 0x2034 in k else None

    def Rx(a):
        c, n = np.cos(a), np.sin(a)
        return np.array([[1, 0, 0], [0, c, -n], [0, n, c]])

    def Ry(a):
        c, n = np.cos(a), np.sin(a)
        return np.array([[c, 0, n], [0, 1, 0], [-n, 0, c]])

    def Rz(a):
        c, n = np.cos(a), np.sin(a)
        return np.array([[c, -n, 0], [n, c, 0], [0, 0, 1]])

    nodes = {}
    for i in range(len(objs)):
        if cls(i) != 'Node':
            continue
        k = kids(i)
        a = u32s(*k[0x2035][0])[1:]
        slots = {a[j]: a[j + 1] for j in range(0, len(a) - 1, 2)}
        parent = u32s(*k[0x960][0])[0]
        p, l = k[0x962][0] if 0x962 in k else (0, 0)
        name = s[p:p + l].decode('utf-16le').rstrip('\0')
        tm, ob = slots[0], slots[1]
        pr = refs(tm)
        M = np.eye(4)
        ok = (cls(tm) == 'Position/Rotation/Scale' and pr and cls(pr[0]) == 'Bezier Position'
              and cls(pr[1]) == 'Euler XYZ' and cls(pr[2]) == 'Bezier Scale')
        off = np.eye(4)
        if ok:
            pos = np.array(f32s(*kids(pr[0])[0x2503][0])[:3])
            ang = [f32s(*kids(j)[0x2501][0])[0] if 0x2501 in kids(j) else 0.0
                   for j in refs(pr[1])]
            sc = np.array(f32s(*kids(pr[2])[0x2505][0])[:3])
            M[:3, :3] = Rz(ang[2]) @ Ry(ang[1]) @ Rx(ang[0]) @ np.diag(sc)
            M[:3, 3] = pos
            if 0x96a in k:
                off[:3, 3] = f32s(*k[0x96a][0])[:3]
            if 0x96c in k:
                off[:3, :3] = np.diag(f32s(*k[0x96c][0])[:3])
        nodes[i] = dict(name=name, parent=parent, ob=ob, M=M, off=off, ok=ok, cls=cls(ob))

    cache = {}

    def world(i):
        if i not in nodes:
            return np.eye(4)
        if i not in cache:
            cache[i] = world(nodes[i]['parent']) @ nodes[i]['M']
        return cache[i]

    partes = []
    for i, n in nodes.items():
        if n['cls'] != 'Editable Mesh' or not n['ok']:
            continue
        k = kids(n['ob'])
        if 0x914 not in k:
            continue
        p, l = k[0x914][0]
        cnt = u32s(p, 4)[0]
        v = np.frombuffer(s[p + 4:p + 4 + cnt * 12], '<f4').reshape(-1, 3).astype(float)
        p, l = k[0x912][0]
        fc = u32s(p, 4)[0]
        fa = np.frombuffer(s[p + 4:p + 4 + fc * 20], '<u4').reshape(-1, 5)
        T = world(i) @ n['off']
        vw = (np.c_[v, np.ones(len(v))] @ T.T)[:, :3]
        partes.append(dict(name=n['name'], group=nodes.get(n['parent'], {}).get('name', ''),
                           V=vw, F=fa[:, :3].astype(int)))
    return partes


if __name__ == '__main__':
    partes = extraer(sys.argv[1])
    pickle.dump(partes, open(sys.argv[2], 'wb'))
    print(len(partes), 'partes ->', sys.argv[2])
