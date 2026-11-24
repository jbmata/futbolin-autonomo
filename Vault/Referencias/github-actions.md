# Entrenamientos en GitHub Actions

Repositorio público, rama `train` (lleva el código; `dev` se pone al día después). Cada workflow se
lanza subiendo un cambio a su fichero de disparo. Ningún workflow escribe en el repositorio: los
resultados se descargan con `gh run download <id> -D <carpeta>`.

| Workflow | Fichero de disparo | Qué hace |
|---|---|---|
| `pipeline.yml` | `.github/entrenar.txt` | Optimiza el profesor, elige con 240 partidos y entrena alumnos |
| `train.yml` | `.github/lanzar-entrenamiento.txt` | DAgger + PPO |
| `optimizar-profesor.yml` | `.github/optimizar-profesor.txt` (1.ª línea: minutos por rol) | CMA-ES por rol y validación |
| `alumnos.yml` | `.github/alumnos.txt` | Alumnos DAgger sin PPO y benchmark de 200 partidos |

Las carpetas de descarga (`Modelo/gh_*`) no se suben a git. Resultados: [[historial-entrenamientos]].
Detalle de uso en `Modelo/README.md`.
