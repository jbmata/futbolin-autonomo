"""Campana partida que salva el puno y transmite el giro desde un NEMA 17. Unidades: mm.

Eje X desde la mesa hacia el motor. Puno X=0..89.5; collarin sobre barra X=-18..-2.
Collarin y salida de eje son metalicos; las dos mitades de campana son imprimibles.
    python acople_puno_nema17.py [--ver]
"""
from collections import Counter
from math import cos, sin, pi, sqrt
from pathlib import Path
from struct import pack_into, unpack_from

from build123d import (Axis, Box, Compound, Cylinder, Plane, Polygon, Pos,
                      RectangleRounded, export_stl, extrude)
import _comun

# shortcut: cotas del modelo de mesa, no del puno real; medir antes de fabricar.
DIAMETRO_BARRA = 14.0
DIAMETRO_PUNO = 32.7
LARGO_PUNO = 89.5
HOLGURA_RADIAL_PUNO = 2.0
RADIO_INTERIOR = DIAMETRO_PUNO/2 + HOLGURA_RADIAL_PUNO
PARED = 6.0
RADIO_EXTERIOR = RADIO_INTERIOR + PARED
X_COLLAR = -18.0
X_BRIDA = -2.0
GROSOR_BRIDA = 6.0
RADIO_BRIDA = 36.0
RADIO_M4 = 29.5
SEPARACION_COLLAR = .5
SEPARACION_CAMPANA = .4
X_FONDO = LARGO_PUNO + 4.0
GROSOR_FONDO = 8.0
X_SALIDA = X_FONDO + GROSOR_FONDO
GROSOR_SALIDA = 4.0
LARGO_EJE = 16.0
DIAMETRO_SALIDA = 8.0
RADIO_TORNILLOS_SALIDA = 16.0
LENGUETA_CENTRADO = 1.8
DIAMETRO_CENTRADO = 2*RADIO_INTERIOR - .4
ANCLAJES_BRIDA = [(RADIO_M4*cos(a),RADIO_M4*sin(a)) for a in (pi/4,3*pi/4,5*pi/4,7*pi/4)]
ANCLAJES_SALIDA = [(RADIO_TORNILLOS_SALIDA*cos(a),RADIO_TORNILLOS_SALIDA*sin(a)) for a in (pi/4,3*pi/4,5*pi/4,7*pi/4)]
ANCLAJES_UNION = [(x,y) for x in (32.,74.) for y in (-RADIO_EXTERIOR-3,RADIO_EXTERIOR+3)]


def cilindro_x(d,x0,x1,y=0,z=0):
    return Pos((x0+x1)/2,y,z)*Cylinder(d/2,x1-x0).rotate(Axis.Y,90)


def cortar_mitad(pieza,superior,separacion):
    z = 100 + separacion/2 if superior else -100-separacion/2
    return pieza & (Pos(60,0,z)*Box(300,250,200))


def collarin(superior):
    perfil = Plane.YZ*RectangleRounded(40,28,5)
    pieza = Pos(X_COLLAR,0,0)*extrude(perfil,amount=X_BRIDA-X_COLLAR)
    pieza += cilindro_x(2*RADIO_BRIDA,X_BRIDA-GROSOR_BRIDA,X_BRIDA)
    pieza += cilindro_x(DIAMETRO_CENTRADO,X_BRIDA,X_BRIDA+LENGUETA_CENTRADO)
    pieza -= cilindro_x(DIAMETRO_BARRA,X_COLLAR-1,X_BRIDA+LENGUETA_CENTRADO+1)
    for y,z in ANCLAJES_BRIDA:
        # Roscas M4 nominales, sin representar el filete.
        pieza -= cilindro_x(4,X_BRIDA-GROSOR_BRIDA-1,X_BRIDA+1,y,z)
    for y in (-13.,13.):
        pieza -= Pos(-13,y,0)*Cylinder(2.15 if superior else 2.,40)
        if superior:pieza -= Pos(-13,y,15)*Cylinder(3.75,10)
    return cortar_mitad(pieza,superior,SEPARACION_COLLAR)


def campana(superior):
    pieza = cilindro_x(2*RADIO_EXTERIOR,X_BRIDA,X_SALIDA)
    pieza += cilindro_x(2*RADIO_BRIDA,X_BRIDA,X_BRIDA+GROSOR_BRIDA)
    pieza -= cilindro_x(2*RADIO_INTERIOR,X_BRIDA-1,X_FONDO)
    pieza -= cilindro_x(16.4,X_FONDO-1,X_SALIDA+1)
    for y,z in ANCLAJES_BRIDA:
        pieza -= cilindro_x(4.4,X_BRIDA-1,X_BRIDA+GROSOR_BRIDA+1,y,z)
    for y,z in ANCLAJES_SALIDA:
        pieza -= cilindro_x(3.4,X_FONDO-1,X_SALIDA+1,y,z)
        # shortcut: alojamiento provisional de inserto termico M3, Ø4.6 x 5; adaptar al inserto elegido.
        pieza -= cilindro_x(4.6,X_SALIDA-5,X_SALIDA+1,y,z)
    pieza = cortar_mitad(pieza,superior,SEPARACION_CAMPANA)
    for x,y in ANCLAJES_UNION:
        z = SEPARACION_CAMPANA/2 if superior else -SEPARACION_CAMPANA/2-6
        oreja = Pos(x,y,z)*extrude(RectangleRounded(12,10,2),amount=6)
        pieza += oreja
        pieza -= Pos(x,y,0)*Cylinder(1.7,20)
        if superior:
            pieza -= Pos(x,y,5.2)*Cylinder(3.25,4)
        else:
            r = 6.0/sqrt(3)
            hexagono = Polygon(*[(x+r*cos(a),y+r*sin(a)) for a in [i*pi/3 for i in range(6)]],align=None)
            pieza -= Pos(0,0,-6.3)*extrude(hexagono,amount=2.8)
    return pieza


def salida_metalica():
    pieza = cilindro_x(42,X_SALIDA,X_SALIDA+GROSOR_SALIDA)
    pieza += cilindro_x(16,X_SALIDA-2,X_SALIDA)
    pieza += cilindro_x(DIAMETRO_SALIDA,X_SALIDA+GROSOR_SALIDA,X_SALIDA+GROSOR_SALIDA+LARGO_EJE)
    for y,z in ANCLAJES_SALIDA:
        pieza -= cilindro_x(3.4,X_SALIDA-1,X_SALIDA+GROSOR_SALIDA+1,y,z)
    return pieza


def piezas():
    return [collarin(True),collarin(False),campana(True),campana(False),salida_metalica()]


def referencias():
    puno = cilindro_x(DIAMETRO_PUNO,0,LARGO_PUNO)
    barra = cilindro_x(DIAMETRO_BARRA,-75,0)
    # Envolvente provisional del acoplamiento comercial 5-8; no es una pieza para imprimir.
    x = X_SALIDA+GROSOR_SALIDA+4
    acople = cilindro_x(20,x,x+25)
    acople -= cilindro_x(8.1,x-1,x+12)
    acople -= cilindro_x(5.1,x+12,x+26)
    return [puno,barra,acople]


def comprobar(lista):
    assert len(lista)==5
    for p in lista:assert p.is_valid and len(p.solids())==1
    for i,a in enumerate(lista):
        for b in lista[i+1:]:
            q=a&b;assert not q or q.volume<1e-5,'Las piezas se solapan'
    puno,barra,acople = referencias()
    for p in lista:
        for r in (puno,barra,acople):
            q=p&r;assert not q or q.volume<1e-5,'Interferencia con puno, barra o acople'
    for y,z in ANCLAJES_BRIDA:
        tornillo = cilindro_x(3.9,X_BRIDA-GROSOR_BRIDA,X_BRIDA+GROSOR_BRIDA,y,z)
        herramienta = cilindro_x(8,X_BRIDA+GROSOR_BRIDA,X_SALIDA+60,y,z)
        for p in lista:
            for r in (tornillo,herramienta):
                q=p&r;assert not q or q.volume<1e-5,'Acceso a un M4 de la brida bloqueado'
    for y,z in ANCLAJES_SALIDA:
        herramienta = cilindro_x(6,X_SALIDA+GROSOR_SALIDA,X_SALIDA+60,y,z)
        for p in lista+[acople]:
            q=p&herramienta;assert not q or q.volume<1e-5,'Acceso a un M3 de salida bloqueado'
    # Comprobar la retirada radial de las mitades alrededor del puno intacto.
    for i in range(4):
        desplazada = Pos(0,0,35 if i%2==0 else -35)*lista[i]
        q=desplazada&puno;assert not q or q.volume<1e-5,'Desmontaje bloqueado por el puno'
    for x,y in ANCLAJES_UNION:
        herramienta = Pos(x,y,19)*Cylinder(3.25,26)
        for p in lista:
            q=p&herramienta;assert not q or q.volume<1e-5,'Acceso a tornillo de cierre bloqueado'
    assert X_FONDO-LARGO_PUNO>=4 and HOLGURA_RADIAL_PUNO>=2
    print('OK: cinco solidos validos, puno libre, centrados, desmontaje radial y acceso a tornillos.',flush=True)


def exportar_stl(pieza,nombre,carpeta,invertir=False):
    orientada = pieza.rotate(Axis.X,180) if invertir else pieza
    orientada = Pos(0,0,-orientada.bounding_box().min.Z)*orientada
    archivo = Path(carpeta)/(nombre+'.stl')
    export_stl(orientada,str(archivo),tolerance=.01,angular_tolerance=.1)
    datos = bytearray(archivo.read_bytes())
    assert len(datos)==84+50*unpack_from('<I',datos,80)[0]
    # Quitar ceros residuales de los giros para que coincidan las aristas en el STL.
    for inicio in range(84,len(datos),50):
        for i in range(inicio+12,inicio+48,4):
            if abs(unpack_from('<f',datos,i)[0])<1e-9:pack_into('<f',datos,i,0.)
    aristas=Counter()
    for inicio in range(84,len(datos),50):
        puntos=[unpack_from('<3f',datos,inicio+i) for i in (12,24,36)]
        aristas.update(tuple(sorted((a,b))) for a,b in zip(puntos,puntos[1:]+puntos[:1]))
    assert all(n==2 for n in aristas.values()),(nombre,'STL sin cerrar')
    archivo.write_bytes(datos)


if __name__=='__main__':
    args = _comun.argumentos(__doc__.splitlines()[0])
    if Path(args.out).resolve()==_comun.CARPETA.resolve():args.out=str(_comun.CARPETA/'acople_puno_nema17')
    lista = piezas();comprobar(lista)
    nombres = ['collarin_superior_metal','collarin_inferior_metal','campana_superior','campana_inferior','salida_eje8_metal']
    for i,(p,nombre) in enumerate(zip(lista,nombres)):
        _comun.exportar(p,nombre,args.out,stl=False)
        exportar_stl(p,nombre,args.out,invertir=i in (1,3))
    _comun.exportar(Compound(lista),'acople_puno_montado',args.out,stl=False)
    _comun.exportar(Compound(lista+referencias()),'acople_puno_referencia',args.out,stl=False)
    if args.ver:_comun.mostrar(*lista,*referencias(),nombres=nombres+['puno de referencia','barra de referencia','acople 5-8 de referencia'],colores=['silver','silver','seagreen','seagreen','silver','dimgray','silver','gold'])
