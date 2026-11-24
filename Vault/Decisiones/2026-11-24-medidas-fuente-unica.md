# Decisión: `Medidas/mesa.json` es la única fuente de medidas de la mesa

- Fecha: 2026-11-24
- Estado: adoptada

## Contexto
El 3D (`Mecanica/3d/futbolin_actualizado`) y el simulador (`Modelo/world.py`) tenían medidas
distintas y cada una se copiaba a mano de la otra. El 3D venía de un modelo genérico (`Foosball.max`)
con los agujeros del plano de futbolinmadrid.com encima, y eso dejaba la mesa descentrada y con la
misma carrera (115,68 mm) en todas las barras. El usuario pidió que el 3D y el simulador coincidan
con los datos reales.

## Decisión
- `Medidas/mesa.json` guarda las medidas, cada una con su fuente: `real`, `plano`, `cad`, `ficha` o `estimado`.
- El simulador la lee al arrancar (`world.load_measurements()`, mesa `FUTBOLIN_MESA=mesa`, por defecto).
- El 3D se ajusta con `Mecanica/3d/futbolin_actualizado/aplicar_medidas.py` (Blender 5.3) y después
  se regeneran el GLB y el STEP.
- `Medidas/comprobar.py` compara barra a barra el 3D, el simulador y `mesa.json`, y sale con error
  si algo no coincide. **Comprobado** 2026-11-24: 48 de 48.
- El 3D de referencia es `futbolin_actualizado` (lo dijo el usuario).

## Por qué (y alternativas descartadas)
- Una sola fuente evita que el 3D y el simulador se separen sin que nadie lo note.
- Descartado: tomar el 3D como base. Sus topes y posiciones son del modelo genérico, no de una mesa real.
- Descartado: copiar las medidas del 3D al simulador (lo que se hizo el 2026-11-23 con la mesa `cad`,
  ya eliminada). Llevaba los errores del modelo genérico al simulador.
- Se mantiene la mesa `sim` solo para evaluar los modelos entrenados antes del 2026-11-23.

Relacionado: [[2026-11-24-topes-y-carrera-por-barra]], [[2026-11-24-referencia-desde-tope-motores]].
