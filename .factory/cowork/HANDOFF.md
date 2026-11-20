<!-- Managed by the Software Factory. Tarjeta de handoff cowork → factoría. -->
# HANDOFF — del cowork a la factoría

> Terminaste el brief con Aura. El brief vive en `.factory/intake/` (dentro del repo).
> Esto es lo único que hace falta para arrancar la construcción.

## Para arrancar la construcción

**Cuando arranques el proyecto con la factoría, da a Claude Code (u opencode) acceso a este directorio
y escribe `guíame` (o `/factory-intake`).**

```bash
# en la carpeta del repo (la que tiene .factory/intake/)
claude            # o tu comando de opencode
```
Dentro del agente, escribe:
```
guíame
```
*(o directamente `/factory-intake`)*

El agente lee el brief de `.factory/intake/`, **valida** que esté bien (criterios medibles, no-goals,
sin secretos), pre-rellena el cuestionario (**no te vuelve a preguntar lo que ya contaste**), define
objetivos y roadmap, y se **detiene en el sign-off** antes de construir. Funciona igual en **Claude Code
y opencode**.

> **Hand the directory to Claude Code (or opencode) and type `guíame`.** That's the whole handoff.

---

## ▶ Sign-off de requisitos (un humano firma antes de construir) ◀

Antes de que la factoría pase de *descubrir* a *construir*, **una persona** (product owner) **aprueba**
el brief — en sesión (modo checkpoint) o vía PR sobre `.factory/intake/**`. La factoría **nunca**
construye sin esta aprobación.

Checklist de aprobación:

- [ ] **Criterios de éxito medibles** — cada criterio es **verificable** por un humano o un test. Los
      vagos ("que sea rápido", "que se vea bien") se rechazan y se reescriben.
- [ ] **No-goals explícitos** — está claro qué queda **fuera** del MVP (acota cuándo parar de construir).
- [ ] **Sin secretos** — ninguna contraseña, token ni clave en los ficheros versionados. Las claves van
      luego en un `.env` git-ignorado, nunca en el intake.

Aprobado el checklist → la factoría sincroniza tareas y **construye** hasta cumplir los criterios.

---

*Detalles del flujo y de cómo se generó el brief: `.factory/cowork/README.md`.*
