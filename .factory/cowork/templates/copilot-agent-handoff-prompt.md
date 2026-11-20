<!-- Managed by the Software Factory. Prompt para el agente de codificación de Copilot (aterrizar el intake). -->
# Prompt para github.com/copilot/agents — aterrizar el brief en el repo

> **Cuándo:** terminaste la sesión con Aura en un **Copilot Space** y quieres que los documentos del
> brief queden en el repo **sin tocar git**. Copia este prompt en `github.com/copilot/agents`,
> selecciona tu repositorio, y pega debajo el contenido que Aura generó.
>
> **Requiere un plan Copilot de pago** (Pro/Pro+/Business/Enterprise) para que el agente abra el PR.
> Con **Copilot Free**, usa el camino de **export manual** (descarga/copia los documentos y déjalos en
> `.factory/intake/`) — ver `.factory/cowork/README.md`, camino (c).

---

En el repositorio seleccionado, crea/actualiza estos ficheros dentro de `.factory/intake/` con
**exactamente** estos nombres y el contenido que pego más abajo, y abre un **Pull Request** en una rama
`intake/<nombre-corto-del-proyecto>`:

- `.factory/intake/00-brief.md`
- `.factory/intake/01-vision-problema.md`
- `.factory/intake/02-usuarios-objetivo.md`
- `.factory/intake/03-catalogo-funciones.md`
- `.factory/intake/04-pantallas-desde-disenos.md`
- `.factory/intake/05-criterios-exito.md`
- `.factory/intake/06-restricciones-integraciones.md`
- `.factory/intake/07-glosario-pendientes.md`
- `.factory/intake/intake.yml`
- `.factory/intake/assets/manifest.yml`

Reglas:
- **No inventes contenido**: usa solo lo que pego. Si algo falta, déjalo como `(pendiente)`.
- **No incluyas secretos** (contraseñas, tokens, claves). Si una integración los necesita, márcalo como
  pendiente — irán luego en un `.env` git-ignorado.
- El título del PR: `intake: <nombre del proyecto>`. En el cuerpo, resume en lenguaje de negocio qué
  incluye el brief y enlaza `.factory/intake/05-criterios-exito.md`.

--- CONTENIDO DEL BRIEF (pega aquí lo que generó Aura) ---
