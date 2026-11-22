// carrovsolt — placa del plano aportado por el usuario. Unidades: mm.
// Origen XY en el centro; cara inferior Z=0. Todos los huecos son pasantes.
// Sin sobremedida de impresion, avellanados ni rebajes no dibujados.

/* [Placa] */
lado = 65.50;
// Espesor confirmado por el usuario; el plano no lo indica.
espesor = 6.00;
// Estimado del dibujo, sin cota de radio.
radio_esquina = 3.00;

/* [Taladros] */
diametro_taladro = 5.10;
diametro_grande = 7.20;
paso = 10.00;

/* [Ranuras] */
ancho_ranura_central = 5.00;
// Longitudes totales, incluidos los extremos semicirculares.
largo_ranura_central = 25.64;
ancho_ranura_lateral = 3.00;
largo_ranura_lateral = 9.50;
// Centro a borde exterior: estimado del dibujo, no acotado.
borde_centro_ranura_lateral = 5.00;

/* [Hidden] */
$fn = 96;

assert(espesor > 0, "El espesor debe ser positivo.");
assert(radio_esquina > 0 && 2 * radio_esquina < lado,
       "Radio de esquina incompatible con el lado.");
assert(largo_ranura_central >= ancho_ranura_central &&
       largo_ranura_lateral >= ancho_ranura_lateral,
       "La longitud total de una ranura debe ser al menos su ancho.");

module ranura_vertical(ancho, largo) {
    hull()
        for (y = [-(largo - ancho) / 2, (largo - ancho) / 2])
            translate([0, y]) circle(d = ancho);
}

module carrovsolt_planta() {
    difference() {
        offset(r = radio_esquina)
            square([lado - 2 * radio_esquina,
                    lado - 2 * radio_esquina], center = true);

        // Reticula centrada de 10 mm, reconstruida del dibujo.
        // Cinco taladros superiores, uno intermedio, tres centrales,
        // uno inferior intermedio y dos inferiores: doce de diametro 5.10.
        for (p = [[-2, 2], [-1, 2], [0, 2], [1, 2], [2, 2],
                  [0, 1], [-1, 0], [0, 0], [1, 0],
                  [0, -1], [-1, -2], [1, -2]])
            translate([p[0] * paso, p[1] * paso])
                circle(d = diametro_taladro);

        // Tres taladros de diametro 7.20 en la fila inferior.
        for (x = [-2 * paso, 0, 2 * paso])
            translate([x, -2 * paso]) circle(d = diametro_grande);

        // Dos ranuras largas de 5.00 x 25.64.
        for (x = [-2 * paso, 2 * paso])
            translate([x, 0])
                ranura_vertical(ancho_ranura_central, largo_ranura_central);

        // Seis ranuras laterales de 3.00 x 9.50.
        for (x = [-(lado / 2 - borde_centro_ranura_lateral),
                    lado / 2 - borde_centro_ranura_lateral])
            for (y = [-2 * paso, 0, 2 * paso])
                translate([x, y])
                    ranura_vertical(ancho_ranura_lateral, largo_ranura_lateral);
    }
}

module carrovsolt() {
    linear_extrude(height = espesor, convexity = 10)
        carrovsolt_planta();
}

carrovsolt();
