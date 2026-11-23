"""Horquilla lineal para un V-Slot 2020; motor exterior derecho. Unidades: mm.

Vista superior: X hacia el recorrido, pared en X=-68, Y derecha e izquierda.
Z es altura; el perfil ocupa X>=0, Y/Z=-10..10. El eje queda a Z=0.
Dos mejillas, dos alas de pared; sin cartela inferior exterior del lado sin motor.
El STEP contiene solo el soporte; el STL apoya la cara de pared en Z=0.
    python soporte_lineal_motor2020.py [--ver]
"""
from pathlib import Path
from build123d import Axis, Box, Cylinder, Plane, Polygon, Pos, RectangleRounded, export_stl, extrude
import _comun
from motor_17hs4401 import DIAMETRO_RESALTE, PASO_MOTOR

ESPESOR = 6.0
HOLGURA_ENCAJE = .2
LARGO_ENCAJE = 50.0
INICIO_PASO_CORREA = -10.0
ALTURA_PASO_CORREA = 14.0
ANCLAJES = (12.0, 32.0)
DIAMETRO_M5 = 5.4
DIAMETRO_M3 = 3.4
DIAMETRO_ACCESO_NEMA = 8.0
X_MOTOR = -32.0
ALTURA_EJE_MOTOR = 0.0
Y_INTERIOR = 10 + HOLGURA_ENCAJE
Y_MEJILLA = Y_INTERIOR
Y_MOTOR = Y_MEJILLA + ESPESOR
X_PARED = -68.0                  # shortcut: pared provisional, adaptar a las medidas reales.
ESPESOR_PARED_SOPORTE = 8.0
ESPESOR_NERVIO = 4.0
Z_BASE_ENCAJE = -10 - HOLGURA_ENCAJE - ESPESOR
Z_BASE = -40.0
Z_SUPERIOR = 40.0
TORNILLOS_PARED = [(y,z) for y in (-36.0,36.0) for z in (-29.5,29.5)]
DIAMETRO_PARED = 6.6


def soporte_lineal_motor():
    inicio = X_PARED + ESPESOR_PARED_SOPORTE - .2
    largo = LARGO_ENCAJE - inicio
    alto = Z_SUPERIOR - Z_BASE
    # Fondo solo bajo el encaje: dejar abierto el canal central hacia la pared.
    pieza = Pos((-2 + LARGO_ENCAJE)/2,0,Z_BASE_ENCAJE+ESPESOR/2) * Box(LARGO_ENCAJE+2,2*Y_MOTOR,ESPESOR)
    for lado in (-1,1):
        pieza += Pos((inicio+LARGO_ENCAJE)/2,lado*(Y_MEJILLA+ESPESOR/2),(Z_BASE+Z_SUPERIOR)/2) * Box(largo,ESPESOR,alto)
        ala = Pos(lado*(Y_MEJILLA+20),(Z_BASE+Z_SUPERIOR)/2) * RectangleRounded(40,alto,3)
        pieza += Pos(X_PARED,0,0) * extrude(Plane.YZ * ala,amount=ESPESOR_PARED_SOPORTE)
    # Dos apoyos laterales dejan pasar los ramales superior e inferior de la correa.
    for lado in (-1,1):
        pieza += Pos(-1,lado*(Y_INTERIOR+4)/2,0)*Box(2,Y_INTERIOR-4,2*Y_INTERIOR)
    triangulo = Polygon((inicio,Y_MOTOR-2),(0,Y_MOTOR-2),(inicio,Y_MEJILLA+40),align=None)
    nervio = extrude(triangulo,amount=ESPESOR_NERVIO)
    for refuerzo in (nervio,nervio.mirror(Plane.XZ)):
        pieza += Pos(0,0,Z_SUPERIOR-ESPESOR_NERVIO)*refuerzo
    pieza += Pos(0,0,Z_BASE)*nervio
    # Abrir por arriba la salida hacia el perfil, conservando los anclajes M5.
    pieza -= Pos((INICIO_PASO_CORREA+LARGO_ENCAJE+1)/2,0,(ALTURA_PASO_CORREA+Z_SUPERIOR+1)/2)*Box(
        LARGO_ENCAJE+1-INICIO_PASO_CORREA,2*(Y_MEJILLA+41),Z_SUPERIOR+1-ALTURA_PASO_CORREA)
    pieza -= Pos((INICIO_PASO_CORREA+LARGO_ENCAJE+1)/2,0,(Z_BASE-1+Z_BASE_ENCAJE)/2)*Box(
        LARGO_ENCAJE+1-INICIO_PASO_CORREA,2*(Y_MEJILLA+41),Z_BASE_ENCAJE-Z_BASE+1)
    rampa = Polygon((INICIO_PASO_CORREA-.2,ALTURA_PASO_CORREA-.2),
                    (20,ALTURA_PASO_CORREA-.2),(INICIO_PASO_CORREA-.2,Z_SUPERIOR),align=None)
    refuerzo = extrude(Plane.XZ*rampa,amount=ESPESOR)
    for r in (refuerzo,refuerzo.mirror(Plane.XY)):
        for y in (-Y_MEJILLA,Y_MOTOR):pieza += Pos(0,y,0)*r
    def taladro_y(d,x,z,y=0,largo=100):
        return Pos(x,y,z)*Cylinder(d/2,largo).rotate(Axis.X,90)
    for x in ANCLAJES:pieza -= taladro_y(DIAMETRO_M5,x,0)
    pieza -= taladro_y(DIAMETRO_RESALTE+.4,X_MOTOR,ALTURA_EJE_MOTOR)
    for dx in (-PASO_MOTOR/2,PASO_MOTOR/2):
        for dz in (-PASO_MOTOR/2,PASO_MOTOR/2):
            pieza -= taladro_y(DIAMETRO_M3,X_MOTOR+dx,ALTURA_EJE_MOTOR+dz,Y_MEJILLA+ESPESOR/2,ESPESOR+2)
            pieza -= taladro_y(DIAMETRO_ACCESO_NEMA,X_MOTOR+dx,ALTURA_EJE_MOTOR+dz,-Y_MEJILLA-ESPESOR/2,ESPESOR+2)
    for y,z in TORNILLOS_PARED:
        pieza -= Pos(X_PARED+ESPESOR_PARED_SOPORTE/2,y,z)*Cylinder(DIAMETRO_PARED/2,ESPESOR_PARED_SOPORTE+2).rotate(Axis.Y,90)
    assert pieza.is_valid and len(pieza.solids())==1
    assert LARGO_ENCAJE>max(ANCLAJES)+DIAMETRO_M5 and HOLGURA_ENCAJE>0
    assert DIAMETRO_ACCESO_NEMA>DIAMETRO_M3
    return pieza


if __name__ == "__main__":
    args=_comun.argumentos(__doc__.splitlines()[0]);pieza=soporte_lineal_motor()
    _comun.exportar(pieza,"soporte_lineal_motor2020",args.out,stl=False)
    imprimible=Pos(0,0,-X_PARED)*pieza.rotate(Axis.Y,-90)
    assert abs(imprimible.bounding_box().min.Z)<1e-6
    export_stl(imprimible,str(Path(args.out)/"soporte_lineal_motor2020.stl"),tolerance=.01,angular_tolerance=.1)
    if args.ver:_comun.mostrar(pieza,nombres=["horquilla lineal motor exterior"],colores=["orange"])
