# Historial de entrenamientos

Goles por minuto (marca − recibe) contra el heurístico rival de nivel 0,7, partidos de 60 s.
La tabla completa y su análisis están en `Modelo/docs/REFERENCIA.md` §6. Todo lo de esta página es
de la mesa **`sim`** (antigua): no se puede comparar con lo que se entrene con `Medidas/mesa.json`.

| Fecha | Run | Qué se probó | Resultado (**Comprobado**) |
|---|---|---|---|
| 2026-11-22 | local `ppo_v4` | PPO 4M pasos desde imitación | No mejoró la imitación |
| 2026-11-22 | 45023586272 | DAgger y DAgger + PPO | Profesor +0,16; alumnos −0,05 a −0,32; PPO siempre peor |
| 2026-11-22 | 45033300387 | Profesor optimizado por roles | +0,37 frente a +0,33, sin diferencia clara |
| 2026-11-23 | 45064712873 | Cadena profesor → alumnos | Profesor `base_90` +0,37; alumnos con cámara ≈ −0,33; con sensores ideales +0,15 |
| 2026-11-23 | 45124636693 | Compensación de latencia | Referencia −0,43 → compensado −0,08 |
| 2026-11-23 | 45142565146 | Semillas y profesor con ojos del alumno | Profesor con cámara −0,06; alumnos −0,08 / −0,30 / −0,05 (media −0,14) |
| 2026-11-23 | 45218541671 / 45218541712 | Profesor y alumnos en una mesa intermedia | **Descartados**: esa mesa ya no existe |

## Conclusiones
- El hueco entre alumno y profesor es de **percepción** (latencia y ruido de la cámara), no de imitación.
- El ruido entre semillas es grande: una sola semilla engaña. Evaluar con 200 partidos e IC 95 %.
- Modelo guardado: `Modelo/models/alumno/compensado_s3.zip` (mejor de 3 semillas, −0,05). Solo vale con
  `FUTBOLIN_MESA=sim`.

Ver [[2026-11-22-dagger-sin-ppo]] y [[2026-11-23-compensacion-latencia]].
