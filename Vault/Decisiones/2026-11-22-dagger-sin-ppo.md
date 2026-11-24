# Decisión: entrenar al alumno con DAgger y no usar PPO

- Fecha: 2026-11-22
- Estado: adoptada

## Contexto
El plan inicial era PPO con actor-crítico asimétrico, arrancando de una imitación del heurístico.

## Decisión
- Profesor: heurístico con parámetros por rol (`RoleParams`), optimizado con CMA-ES (`optimize_teacher.py`).
  El que se usa es `Modelo/models/teacher/base_90.json`.
- Alumno: red que imita al profesor con DAgger (`dagger.py`), viendo lo que vería el robot (cámara y encoders).
- PPO no se usa para el modelo final.

## Por qué (y alternativas descartadas)
- **Comprobado** en GitHub (runs 45023586272 y 45064712873): DAgger da un alumno razonable y PPO lo
  empeora siempre (olvida lo imitado), tanto con poco ruido como con ruido normal.
- PPO desde cero no descubre la secuencia armar-golpear.
- Abierto, sin decidir: RL anclado al alumno con una pérdida de imitación.

Ver [[historial-entrenamientos]].
