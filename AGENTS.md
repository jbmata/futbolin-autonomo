# Instrucciones para agentes (Claude Code, Codex, otros)

Este archivo es la **fuente única**. `CLAUDE.md` solo lo importa. Edita aquí.

## Qué es el proyecto
Robot que juega al futbolín **madrileño** contra una persona: maneja las cuatro barras de un equipo con
ocho motores paso a paso (NEMA17 17HS4401 + TMC2209, placa propia con STM32H733 en `Hardware/MCU-RT`) y ve
la mesa con una cámara Logitech Brio. El agente se entrena en un simulador propio (`Modelo/`, Python) y las
medidas de la mesa están en `Medidas/mesa.json`, la fuente única del simulador y del 3D (`Mecanica/3d`).
Rama de trabajo: `train`. Estado actual en `Vault/Estado.md`.

## Memoria a largo plazo: LEE ESTO AL EMPEZAR
La memoria del proyecto vive en `Vault/` (Markdown, formato Obsidian).
1. Lee `Vault/Index.md`, luego `Vault/Estado.md` y `Vault/Pendientes.md`.
2. Para saber dónde está cada cosa, usa `Vault/Estructura.md`.
3. Antes de cambiar algo ya decidido, busca en `Vault/Decisiones/`.
4. **Si una decisión nueva contradice una anterior**: no la borres ni la reescribas, y no decidas tú en silencio.
   Avisa al usuario de la contradicción. Si confirma el cambio, crea la decisión nueva con `Sustituye a: [[anterior]]`,
   marca la anterior como `Estado: sustituida por [[nueva]]` (el resto del archivo queda como estaba, es historia)
   y actualiza `Estado.md` / `Pendientes.md` y los apartados de `Referencias/` que digan lo contrario.

### Reglas de escritura
- **Documenta siempre**: todo lo que hagas o descubras queda en `Vault/` antes de dar la tarea por terminada,
  también si queda a medias, falla o el usuario te interrumpe (anota en qué estado deja los ficheros).
  No esperes a que te lo pidan ni al final de la sesión.
- **Una sola memoria**: guarda todo lo que aprendas en `Vault/`, no en la memoria privada de tu herramienta. Así Claude, Codex y cualquier otro agente trabajan sobre el mismo contexto.
- Un hecho o decisión por archivo. Si existe, **actualízalo**; no dupliques.
- Fechas absolutas (AAAA-MM-DD). Nunca "ayer" ni "la semana pasada".
- Guarda el **porqué** de las decisiones, no lo que ya está en el código o en git.
- Etiqueta la procedencia: **Comprobado** / **Fuente externa** / **Supuesto**.
- Al terminar una sesión con cambios relevantes: añade `Vault/Diario/AAAA-MM-DD.md`
  y actualiza `Estado.md` / `Pendientes.md`.
- La vault **enlaza** la documentación técnica existente; no la copia.
- Nunca copies a la vault credenciales ni archivos `*.local.*`.
