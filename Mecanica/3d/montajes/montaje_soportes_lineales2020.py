"""Montaje de referencia: horquilla, un perfil y motor exterior derecho. Unidades: mm.

X sigue el perfil, Y derecha/izquierda y Z altura. Poleas a Z=0, correa centrada a Y=0.
Los ramales recorren las ranuras superior e inferior del 2020.
El tensor tiene dos mejillas unidas por la punta y dos separadores de rodamiento de 5.2 mm.
Poleas, perfil y motor son referencias, no piezas para imprimir.
    python montaje_soportes_lineales2020.py [--ver]
"""
from pathlib import Path
from math import pi, sqrt
from build123d import Axis, Box, Compound, Cylinder, Face, Line, Plane, Pos, ThreePointArc, Wire, extrude
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent.parent / 'piezas'))
import _comun
from soporte_lineal_motor2020 import (ALTURA_EJE_MOTOR,ALTURA_PASO_CORREA,ANCLAJES,ESPESOR,ESPESOR_PARED_SOPORTE,
                                     INICIO_PASO_CORREA,Z_SUPERIOR,
                                     HOLGURA_ENCAJE,LARGO_ENCAJE,TORNILLOS_PARED,X_MOTOR,X_PARED,
                                     Y_INTERIOR,Y_MEJILLA,Y_MOTOR,soporte_lineal_motor)
from soporte_lineal_tensor2020 import RECORRIDO,SEMILUZ,X_POLEA,PUNTA_INTERIOR,soporte_lineal_tensor
from motor_17hs4401 import ALTURA_EJE,LARGO_MOTOR,PASO_MOTOR,X_CARA_MOTOR,motor_referencia
from vslot2020 import vslot2020

LARGO=300.0                     # ejemplo; la longitud real no esta definida
Y_CORREA=0.0
Y_TENSOR=-SEMILUZ
ANCHO_CORREA=6.0
ANCHO_LOCA=10.0
SEPARADOR=Y_CORREA-ANCHO_LOCA/2-Y_TENSOR


def _cilindro_y(d,largo,x,y,z=ALTURA_EJE_MOTOR):
    return Pos(x,y,z)*Cylinder(d/2,largo).rotate(Axis.X,90)


def correa_referencia():
    # Envolvente sin dientes; el radio primitivo de 20T es 20*2/(2*pi).
    x1,x2=X_MOTOR,LARGO+X_POLEA
    r1,r2=20*2/(2*pi),8+.7
    c=(r1-r2)/(x2-x1);s=sqrt(1-c*c)
    def borde(e):
        a,b=r1+e,r2+e
        p1=(x1+a*c,a*s);p2=(x2+b*c,b*s)
        p3=(x2+b*c,-b*s);p4=(x1+a*c,-a*s)
        return Wire([Line(p1,p2),ThreePointArc(p2,(x2+b,0),p3),
                     Line(p3,p4),ThreePointArc(p4,(x1-a,0),p1)])
    seccion=Face(borde(.7),[borde(-.7)])
    correa=Pos(0,Y_CORREA+ANCHO_CORREA/2,ALTURA_EJE_MOTOR)*extrude(Plane.XZ*seccion,amount=ANCHO_CORREA,dir=(0,-1,0))
    assert correa.is_valid and len(correa.solids())==1
    assert abs(correa.bounding_box().min.Y-Y_CORREA+ANCHO_CORREA/2)<1e-6
    assert abs(correa.bounding_box().max.Y-Y_CORREA-ANCHO_CORREA/2)<1e-6
    return correa


def montaje():
    perfil=Pos(LARGO/2,0,0)*vslot2020(LARGO).rotate(Axis.Y,90)
    soporte=soporte_lineal_motor()
    tensor=Pos(LARGO,Y_TENSOR,0)*soporte_lineal_tensor().rotate(Axis.X,90)
    motor=Pos(X_MOTOR,Y_MOTOR,ALTURA_EJE_MOTOR)*(Pos(-X_CARA_MOTOR,0,-ALTURA_EJE)*motor_referencia()).rotate(Axis.Z,90)
    motriz=_cilindro_y(12.2,7,X_MOTOR,Y_CORREA)
    for y in (Y_CORREA-4,Y_CORREA+4):motriz+=_cilindro_y(18,1,X_MOTOR,y)
    motriz+=_cilindro_y(16,6,X_MOTOR,Y_CORREA+7.5)
    motriz-=_cilindro_y(5.1,40,X_MOTOR,Y_CORREA)
    x=LARGO+X_POLEA;loca=_cilindro_y(16,ANCHO_LOCA-2,x,Y_CORREA)
    for y in (Y_CORREA-4.5,Y_CORREA+4.5):loca+=_cilindro_y(18,1,x,y)
    loca-=_cilindro_y(5,ANCHO_LOCA+2,x,Y_CORREA)
    separadores=[]
    for y in (-ANCHO_LOCA/2-SEPARADOR/2,ANCHO_LOCA/2+SEPARADOR/2):
        anillo=_cilindro_y(8,SEPARADOR,x,y)
        anillo-=_cilindro_y(5.2,SEPARADOR+2,x,y)
        separadores.append(anillo)
    separador=Compound(separadores)
    return [perfil,soporte,tensor,motor,motriz,loca,separador,correa_referencia()]


def comprobar(p):
    assert len(p)==8 and SEPARADOR>0 and abs(p[3].bounding_box().max.Y-Y_MOTOR-LARGO_MOTOR)<1e-6
    for i,pieza in enumerate(p):assert pieza.is_valid and len(pieza.solids())==(2 if i==6 else 1)
    for i,a in enumerate(p):
        for j,b in enumerate(p[i+1:],i+1):
            if j==7 and i in (4,5):continue  # Contacto de correa con las poleas simplificadas.
            q=a&b;assert not q or q.volume<1e-5,(i,j,q.volume)
    for dy in (-.05,.05):
        for a in p[:3]:
            q=a&(Pos(0,dy,0)*p[7]);assert not q or q.volume<1e-5,"Paso superior/inferior de correa bloqueado"
    salida=Pos((INICIO_PASO_CORREA+LARGO_ENCAJE)/2,0,(ALTURA_PASO_CORREA+Z_SUPERIOR)/2)*Box(
        LARGO_ENCAJE-INICIO_PASO_CORREA,2*Y_MEJILLA-.2,Z_SUPERIOR-ALTURA_PASO_CORREA)
    q=p[1]&salida;assert not q or q.volume<1e-5,"Escotadura de salida bloqueada"
    canal=Pos((X_PARED+ESPESOR_PARED_SOPORTE-2)/2,0,-13.2)*Box(
        -2-X_PARED-ESPESOR_PARED_SOPORTE,2*Y_MEJILLA-.2,ESPESOR-.2)
    q=p[1]&canal;assert not q or q.volume<1e-5,"Fondo central bloquea el canal"
    q=p[1]&(Pos(25,0,0)*Box(50,20,20));assert not q or q.volume<1e-5,"Encaje del 2020 ocupado"
    for y in (-Y_MEJILLA-16,Y_MEJILLA+16):
        testigo=Pos(-40,y,-38)*Box(2,2,2)
        q=p[1]&testigo
        if y<0:assert not q or q.volume<1e-5,"Cartela inferior izquierda sin retirar"
        else:assert q and abs(q.volume-testigo.volume)<1e-5,"Falta la cartela del lado motor"
    for y in (-Y_MEJILLA-ESPESOR/2,Y_MEJILLA+ESPESOR/2):
        for z in (-20,20):
            testigo=Pos(0,y,z)*Box(2,ESPESOR-.2,2)
            q=p[1]&testigo;assert q and abs(q.volume-testigo.volume)<1e-5,"Falta un nervio frontal superior o inferior"
    for y in (-Y_MEJILLA-ESPESOR/2,Y_MEJILLA+ESPESOR/2):
        q=p[1]&_cilindro_y(22.3,ESPESOR,X_MOTOR,y);assert not q or q.volume<1e-5,"Agujero central bloqueado"
    for x in ANCLAJES:
        for y in (-Y_INTERIOR-ESPESOR/2,Y_INTERIOR+ESPESOR/2):
            tornillo=_cilindro_y(5,10,x,y,0)
            for a in p[:2]:
                q=a&tornillo;assert not q or q.volume<1e-5
            herramienta=_cilindro_y(10,20,x,y/abs(y)*(Y_INTERIOR+ESPESOR+10),0)
            q=p[1]&herramienta;assert not q or q.volume<1e-5,"Acceso M5 bloqueado"
    for dx in (-PASO_MOTOR/2,PASO_MOTOR/2):
        for dz in (-PASO_MOTOR/2,PASO_MOTOR/2):
            x=X_MOTOR+dx;z=ALTURA_EJE_MOTOR+dz
            q=p[1]&_cilindro_y(3,10,x,Y_MEJILLA+ESPESOR/2,z);assert not q or q.volume<1e-5
            herramienta=_cilindro_y(7,50.2,x,-14.9,z)
            for a in p:
                q=a&herramienta;assert not q or q.volume<1e-5,"Acceso al tornillo NEMA bloqueado"
    for y,z in TORNILLOS_PARED:
        q=p[1]&(Pos(X_PARED+4,y,z)*Cylinder(3,12).rotate(Axis.Y,90));assert not q or q.volume<1e-5
        herramienta=Pos(X_PARED+ESPESOR_PARED_SOPORTE+45,y,z)*Cylinder(5.5,90).rotate(Axis.Y,90)
        for a in p[:2]+p[3:5]:
            q=a&herramienta;assert not q or q.volume<1e-5,"Acceso M6 bloqueado"
    for d in (-RECORRIDO/2,0,RECORRIDO/2):
        q=p[2]&_cilindro_y(5,40,LARGO+X_POLEA+d,0);assert not q or q.volume<1e-5
        for a in p[5:7]:
            for b in p[:3]:
                q=(Pos(d,0,0)*a)&b;assert not q or q.volume<1e-5
    puente=Pos(LARGO+PUNTA_INTERIOR+ESPESOR/2,0,0)*Box(ESPESOR-.2,2*(SEMILUZ+ESPESOR)-.2,20)
    q=p[2]&puente;assert q and abs(q.volume-puente.volume)<1e-5,'Falta la union frontal de las dos mejillas'
    for lado in (-1,1):
        for x in (-32,-12):
            herramienta=_cilindro_y(10,20,LARGO+x,lado*(SEMILUZ+ESPESOR+10),0)
            q=p[2]&herramienta;assert not q or q.volume<1e-5,'Acceso a los anclajes del tensor bloqueado'
    q=(Pos(-.5,0,0)*p[0])&p[1];assert q and q.volume>1
    assert LARGO_ENCAJE<LARGO-42
    assert ALTURA_EJE_MOTOR==0 and Y_CORREA==0 and Y_MEJILLA==Y_INTERIOR
    print("OK: tensor doble unido en punta, polea y correa libres en todo el tensado, ejes Z=0 y anclajes accesibles.")


if __name__=="__main__":
    args=_comun.argumentos(__doc__.splitlines()[0])
    piezas=montaje();comprobar(piezas)
    _comun.exportar(Compound(piezas),"montaje_soportes_lineales2020",args.out,stl=False)
    if args.ver:_comun.mostrar(*piezas,nombres=["un perfil 2020","soporte motor","tensor doble","motor exterior","motriz","loca","dos separadores 5.2mm","correa superior/inferior"],colores=["silver","orange","royalblue","dimgray","gold","gold","silver","seagreen"])
