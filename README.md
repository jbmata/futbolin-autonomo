# Futbolín autónomo

Proyecto de investigación para desarrollar un robot que juegue al futbolín **madrileño**
contra una persona, con un agente entrenado en simulación y preparado para pasar a la mesa real.

## Componentes

- [Modelo](./Modelo/README.md): simulador físico del sistema real (motores paso a paso,
  cámara, figuras con taco), entorno Gymnasium, profesor heurístico, imitación con DAgger,
  PPO y evaluación con intervalo de confianza. La memoria técnica (valores y su fuente,
  resultados, lecciones y trabajo futuro) está en [`Modelo/docs/REFERENCIA.md`](./Modelo/docs/REFERENCIA.md).
- [Hardware/MCU-RT](./Hardware/MCU-RT/README.md): esquema KiCad con STM32H733ZGT6,
  ocho drivers TMC2209, ocho puertos AS5600 mediante TCA9548A y cuatro finales
  mecánicos, con instrucciones de montaje y validación.
- `Mecanica/3d/`: piezas 3D en [build123d](https://github.com/gumyr/build123d) (un `.py` por pieza: `carrovsolt.py`, `carrovsolt_nema17.py`,
  `motor_17hs4401.py`, `vslot2020.py` y `montaje.py`). Cada uno genera su `.step`/`.stl` y admite `--ver` (OCP CAD Viewer).

## Cómo empezar

```bash
cd Modelo
pip install -r requirements.txt
python visualize.py --teacher models/teacher/base_90.json     # un partido en el navegador
python -m unittest test_training                              # pruebas
```

## Ramas y entrenamiento en GitHub

- `dev` y `train` apuntan al mismo código. En `train` hay workflows que **solo se lanzan
  al cambiar un fichero de disparo** (`.github/entrenar.txt`, `.github/lanzar-entrenamiento.txt`
  o `.github/optimizar-profesor.txt`), así los entrenamientos no ensucian el historial de `dev`.
- Los resultados (modelos, métricas, paneles y partidos en HTML) salen como artefactos de
  cada ejecución y no se escriben en el repositorio.

## Estado

El simulador todavía no controla la electrónica directamente. Los parámetros de motores,
cámara y mecánica son estimaciones; el diseño electrónico requiere validación eléctrica y
mecánica antes de fabricar o alimentar. Los modelos entrenados dependen de esas estimaciones
y habrá que repetirlos cuando se midan.
