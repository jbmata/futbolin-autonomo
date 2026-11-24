# Estructura del proyecto

_Última revisión: 2026-11-24_

| Carpeta | Qué contiene | Documento de referencia |
|---|---|---|
| `Medidas/` | `mesa.json` (medidas de la mesa, fuente única), `comprobar.py` (3D = simulador = mesa.json) | `Medidas/README.md` |
| `Modelo/` | Simulador (física, motores, cámara), entorno Gymnasium, profesor heurístico, DAgger, PPO, benchmark, visor | `Modelo/README.md`, `Modelo/docs/REFERENCIA.md` |
| `Modelo/models/` | Modelos guardados (profesor `teacher/base_90.json`, alumno `alumno/compensado_s3.zip`). Todos de la mesa `sim` | [[historial-entrenamientos]] |
| `Mecanica/3d/piezas/`, `montajes/`, `referencias/` | Piezas en build123d, montajes de comprobación y STEP de terceros | [[2026-11-24-organizacion-carpeta-3d]] |
| `Mecanica/3d/futbolin_actualizado/` | Mesa completa en Blender: `.blend`, `.glb`, `.step`; `aplicar_medidas.py` la ajusta a `mesa.json` | `Medidas/README.md` |
| `Mecanica/3d/futbolin_con_motores/` | Actuadores del robot sobre la mesa (a medio actualizar) | [[montaje-con-motores]] |
| `Hardware/MCU-RT/` | KiCad: STM32H733, TMC2209, AS5600, finales | `Hardware/MCU-RT/README.md` |
| `Firmware/Control/` | Vacía | — |
| `.github/workflows/` | Entrenamientos en GitHub Actions (rama `train`) | [[github-actions]] |
| `Vault/` | Esta memoria | [[Index]] |

## Archivos .md del proyecto
- `README.md` — visión general y componentes.
- `AGENTS.md` / `CLAUDE.md` — instrucciones para agentes (CLAUDE.md solo importa AGENTS.md).
- `Medidas/README.md` — medidas, cómo se colocan las barras, qué medir y cómo actualizar 3D y simulador.
- `Modelo/README.md` — uso del simulador, workflows y variables de entorno.
- `Modelo/docs/REFERENCIA.md` — memoria técnica: valores con su fuente, resultados (§6), lecciones, trabajo futuro.
- `Modelo/logs/seed_comparison/report.md` — comparación de semillas antigua.
- `Hardware/MCU-RT/README.md` — placa: conexiones, montaje y validación.

## Carpetas de datos que no se suben a git
`Modelo/gh_artifacts/`, `Modelo/gh_pipeline/`, `Modelo/gh_alumnos/`, `Modelo/gh_alumnos2/`
(artefactos descargados de GitHub), `Modelo/logs*`, `Modelo/eval_out/`, `Modelo/.venv/`.

## Archivos sensibles (nunca a la vault ni a commits)
No hay credenciales en el repositorio. No subir `*.local.*` ni `.claude/settings.local.json`.
