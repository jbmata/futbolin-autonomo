# Pendientes

_Última actualización: 2026-11-24_

## Medir en la mesa real (ver `Medidas/README.md`)
- [ ] Distancia entre guías (cara interior de los casquillos). Es la que más cambia las carreras.
- [ ] Ancho de la figura a lo largo de la barra, donde la atraviesa.
- [ ] Diámetro del pasador del portero.
- [ ] Carrera de cada barra de tope a tope, para comprobar la calculada.
- [ ] Ancho de la portería, posición de cada barra desde la línea central, figura y bola.
- [ ] Latencia real de la Logitech Brio.
- [ ] Masas: carro, motor de giro, campana, acople, varilla y figuras.

## 3D
- [x] Regenerar `futbolin_actualizado/futbolin_madrid.step` y pasar `Medidas/comprobar.py` (2026-11-24: 48/48).
- [ ] Rehacer el render y el vídeo de `futbolin_actualizado/` (son de la mesa anterior).
- [ ] **Decidir (usuario)** cómo encaja el actuador con los topes reales ([[montaje-con-motores]]):
      limitar el recorrido del robot a lo que cabe, rediseñar el soporte de pared y el collarín,
      o cambiar los topes de la mesa.
- [x] `futbolin_con_motores/`: `piezas.json` coherente con `guias.json`, `montar_blender.py` anota los
      conflictos en lugar de fallar, montaje y PNG regenerados, `original/` borrada y `LEEME.txt` al día (2026-11-24).
- [ ] Ajustar `futbolin_con_motores/` cuando se tome la decisión anterior.
- [ ] Regenerar el vídeo de `futbolin_con_motores/video/` (es de la mesa anterior).
- [x] Descartar el cambio de `futbolin_con_motores/soporte_pared_rebajado.step` (solo cambiaba la fecha) (2026-11-24).
- [x] Actualizar el README raíz, `LEEME.txt` del montaje, `Medidas/README.md` y las rutas de la vault a la
      organización nueva de `Mecanica/3d` (2026-11-24).

## Simulador y entrenamiento
- [x] Pasar la batería completa de pruebas con la mesa nueva (2026-11-24: 57 OK).
- [x] Commit y push del trabajo del 2026-11-24 (`9f2cd35`). Quedan sin subir `Hardware/` y los borrados de `futbolin3d/`.
- [ ] Reoptimizar el profesor y reentrenar el alumno con la mesa de `mesa.json`.
- [ ] Polea del motor lineal: el montaje usa GT2 de 20 dientes y `hardware.py` sigue con 30.
      **Tema aparcado por el usuario: no tocar sin que lo pida** ([[motores-tema-aparcado]]).
- [ ] Ideas para mejorar el alumno, sin decidir: optimizar el profesor con la vista de la cámara,
      promediar varios alumnos, RL anclado a la imitación, medir el efecto de la latencia y los fps.

## Hardware y otros
- [x] Restaurar C1033 y comprobar sus conexiones (2026-11-24; [[2026-11-24-condensador-reset]]).
- [x] Iniciar PCB con 210 huellas, compactar a 160 × 140 mm y limitarla a cuatro capas
      (2026-11-24; [[2026-11-24-pcb-cuatro-capas-jlcpcb]]).
- [ ] Confirmar dimensiones mecánicas, espacio para cables/disipadores y taladros de fijación.
- [ ] Cerrar el apilado estándar JLCPCB y el dimensionado de potencia, retornos y disipación.
- [ ] Resolver protección ESD del USB, parada de emergencia y frenado regenerativo; validar
      protección de ramas sin fusibles locales. Ver limitaciones en el README de MCU-RT.
- [ ] Revisar colocación de desacoplo, cristal, RESET y comunicaciones; completar las **499
      conexiones sin rutear** y los planos de cobre. No superar **cuatro capas**.
- [ ] Corregir los **363 avisos de serigrafía** y cerrar DRC sin conexiones pendientes antes de fabricar.
- [ ] Validar componentes reales, conectores, carga USB/Pi, AS5600, finales de carrera a 3,3 V
      y condiciones térmicas; implementar y probar firmware y homing.
- [ ] Generar Gerbers, taladros y documentación final; revisión DFM y cotización de JLCPCB
      con cantidad, opciones y envío. No hay precio ni pedido confirmados. Ver [[hardware-mcu-rt]].
- [ ] Poner la rama `dev` al día con `train` cuando el usuario lo pida.
