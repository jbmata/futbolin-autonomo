<!-- Managed by the Software Factory. Plantilla de cowork (toma de requisitos). -->
# Cowork ↔ Factoría — toma de requisitos para no técnicos

> Este bundle es la **plantilla de cowork**: un espacio guiado donde una persona **no técnica**
> describe la plataforma que quiere (subiendo diseños, respondiendo una pregunta a la vez) y produce
> un **brief construible**. Ese brief se deposita en `.factory/intake/` y la factoría lo construye.
> No necesitas saber programar ni tener un PM o un ingeniero delante. Te guía **Aura**.

---

## La idea de los dos mundos

Hay **dos mundos** y un **puente de ficheros** entre ellos:

```
MUNDO 1 — COWORK (requisitos)         PUENTE              MUNDO 2 — FACTORÍA (build)
┌─────────────────────────────┐                     ┌──────────────────────────────┐
│ Tú + Aura, sin jerga      │  .factory/intake/   │ Claude Code / opencode       │
│ · cuentas qué quieres       │ ──(el repo git)──▶  │ /factory-intake → init →     │
│ · subes diseños/capturas    │  brief + intake.yml │ discover → SIGN-OFF → build  │
│ · 1 pregunta por turno      │  + assets/          │ hasta tener el MVP corriendo │
└─────────────────────────────┘                     └──────────────────────────────┘
```

- **Mundo 1 — Cowork (requisitos):** describes *qué* quieres construir. Barato, sin código, a tu ritmo.
- **Mundo 2 — Factoría (build):** Claude Code (u opencode) toma esos requisitos y construye el MVP.

El puente es **el propio repositorio**: el cowork escribe ficheros markdown en `.factory/intake/`, y la
factoría los lee. Sin API, sin formato intermedio — solo ficheros en una carpeta compartida.

> **Why two worlds?** Requirements are cheap, non-technical and human; the build is high-skill and
> high-margin. Keeping them separate lets a non-technical person do all the requirements work alone,
> then hand a clean, signed-off brief to the factory. Same repo, same account, zero friction.

---

## Qué hay en este bundle

```
.factory/cowork/
  PROJECT-INSTRUCTIONS.md      # el "system prompt" de Aura — pégalo como Instructions
  README.md                    # esto que estás leyendo
  HANDOFF.md                   # la tarjeta de una línea para arrancar la factoría + nota de sign-off
  templates/
    00-brief.md … 07-glosario-pendientes.md   # los artefactos del brief (Aura los rellena)
    intake.yml                 # forma "máquina" del cuestionario (para no preguntarte dos veces)
    assets/manifest.yml        # índice de los diseños/capturas que subas
```

`PROJECT-INSTRUCTIONS.md` es **una sola pieza** que sirve en las tres superficies de abajo. Lo demás son
las **plantillas** que Aura rellena con tus respuestas.

---

## Cómo usar `PROJECT-INSTRUCTIONS.md`

Tienes **tres caminos**. El **(a) Claude Cowork** es el camino principal (escribe directo al repo). El
**(b) claude.ai web** es excelente para redactar y subir diseños, pero requiere **exportar a mano**. El
**(c) GitHub Copilot App (Spaces)** es la UI amable de GitHub: Aura vive en un *Space* y el brief
aterriza vía un PR del agente de codificación (o export, en plan Free).

### (a) Claude Cowork — camino primario (escribe directo al repo)

Claude Cowork es la app de escritorio de Anthropic para perfiles no técnicos. Trabaja sobre **carpetas
locales** y puede **escribir ficheros** en ellas. Por eso es el camino recomendado.

1. **Clona el repo** a tu equipo (o pide a quien lleva el proyecto la **copia de trabajo** del repo).
2. En Claude Cowork, **crea un Project local**.
3. **Pon la carpeta del Project = la carpeta del repo** (la copia de trabajo). Así Aura escribe
   directamente en `.factory/intake/`, dentro del repo.
4. Abre `.factory/cowork/PROJECT-INSTRUCTIONS.md`, **copia todo su contenido** y **pégalo como las
   *Instructions*** del Project.
5. **Empieza a hablar con Aura.** Te saluda, te ofrece un modo de entrada (*Guiada* · *Pego lo que ya
   tengo* · *Adivina por mí*) y te lleva una pregunta a la vez. Sube tus diseños/capturas cuando los pida.
6. Al terminar, Aura **escribe el brief** (`00-brief.md` … `07-…`, `intake.yml`, `assets/`) en
   `.factory/intake/`. **Commitea** (idealmente en una rama `intake/<nombre>` y abre un PR — facilita el
   sign-off).

> 💡 La carpeta del Project **es** el repo. No hay export: lo que Aura escribe ya está en su sitio.

### (b) claude.ai web Project — redacción + subida de diseños (export manual)

Los *Projects* de **claude.ai en la web** son geniales para **redactar** requisitos y **subir capturas**
(la visión de Opus las "lee"). Pero su **conector de GitHub es de solo lectura**: puede *traer* ficheros
del repo al conocimiento del Project, pero **no escribe** en el repo. Por eso aquí toca **exportar a mano**.

1. En claude.ai, **crea un Project**.
2. Pega el contenido de `PROJECT-INSTRUCTIONS.md` en las **custom instructions** del Project.
3. **Sube tus diseños/capturas** (PNG, JPG, PDF) al conocimiento del Project.
4. Habla con Aura igual que en Cowork: una pregunta por turno, modo a elegir, confirmando los diseños.
5. Al terminar, **EXPORTA** los artefactos que Aura genera (`00-brief.md` … `07-…`, `intake.yml`,
   `assets/manifest.yml`) — cópialos del chat o descárgalos.
6. **Deposítalos en `.factory/intake/`** de tu copia local del repo (respetando los nombres exactos) y
   **commitea** (rama `intake/<nombre>` + PR recomendado).

> ⚠️ **El conector web es de solo lectura.** La web *no* puede escribir en el repo por sí sola: el paso
> de **exportar + soltar en `.factory/intake/`** lo haces tú. (En Cowork ese paso desaparece.)

> 🔒 **Nunca** pegues contraseñas, tokens ni claves en los artefactos. Si una integración necesita una
> clave, Aura la marca como *pendiente* — la clave se pondrá luego en un `.env` git-ignorado, jamás
> en un fichero versionado.

### (c) GitHub Copilot App (Copilot Spaces) — UI amable de GitHub

GitHub Copilot ofrece una UI web amable para no técnicos: un **Space** (`github.com/copilot/spaces`)
donde pegas las instrucciones de Aura y adjuntas el repo + las plantillas como contexto.

1. Crea un **Space** en `github.com/copilot/spaces` y ponle nombre.
2. Pega TODO `PROJECT-INSTRUCTIONS.md` en el campo de **instrucciones** del Space.
3. Adjunta como fuentes el **repo** y la carpeta `.factory/cowork/templates/`.
4. Habla con Aura igual que en Cowork (una pregunta por turno; sube diseños).
5. Al terminar, aterriza el brief en `.factory/intake/`:
   - **(c) recomendado, sin git:** usa `templates/copilot-agent-handoff-prompt.md` en
     `github.com/copilot/agents` → el agente crea los ficheros y **abre un PR** (necesita plan de pago).
   - **(c-fallback, Free):** **descarga/copia** los artefactos y déjalos en `.factory/intake/` (como el
     camino (b) de la web), respetando los nombres exactos.

> El PR del agente **es** el vehículo de sign-off (ver `HANDOFF.md`). Con **Copilot Free** no hay PR
> automático: usa el export manual.

---

## El flujo de punta a punta

Una vez el brief está en `.factory/intake/`, esto es lo que pasa (el grueso lo hace el agente):

| Paso | Quién | Qué ocurre |
|---|---|---|
| 1. **Intake** | Tú + Aura (cowork) | Brief en `.factory/intake/` (criterios **medibles**, no-goals, diseños). |
| 2. **`/factory-intake`** | Agente | Valida el esquema (criterios medibles, sin secretos, no-goals) e ingiere el brief. |
| 3. **`factory-init`** | Agente | Pre-rellena el cuestionario desde `intake.yml` (**no te vuelve a preguntar**) y escribe `FACTORY.md` + `.factory/config.yml`. |
| 4. **`factory-discover`** | Agente | Profundiza objetivos y siembra el roadmap. |
| 5. **▶ SIGN-OFF ◀** | **Una persona** | Un humano **aprueba** criterios medibles + no-goals + sin secretos. La factoría **no construye** sin esto. |
| 6. **build** | Agente | `/factory-sync` (tareas como Issues) → `/factory-build` (construye por fases con gate `docker compose`) hasta cumplir los criterios. |

**Para arrancar el mundo 2**: da a Claude Code (u opencode) acceso a este directorio y escribe
**`guíame`** (o `/factory-intake`). Eso es todo — el resto está en `HANDOFF.md`.

---

## Nota dual-tool (lado factoría)

La factoría funciona **igual en Claude Code y en opencode**. Los comandos (`/factory-intake`,
`/factory-init`, `/factory-discover`, `/factory-build`…) están instalados para los dos. No tienes que
elegir: usa el agente que prefiera quien lleve el build. El brief que produce el cowork es el mismo para
ambos.

> **Dual-tool, one brief.** Whatever the build operator runs — Claude Code or opencode — reads the exact
> same intake from `.factory/intake/`. The cowork side stays tool-agnostic.

---

## En una frase

**Cuéntale a Aura qué quieres → ella escribe el brief en `.factory/intake/` → un humano lo firma →
la factoría lo construye.** 🚀
