# 04 · Pantallas desde diseños — {{NOMBRE_PROYECTO}}

> Etapa 5 del intake. Aura analiza cada diseño o captura subida siguiendo el bucle
> Leer → Inferir → Confirmar → Enlazar. Nunca se da nada por bueno sin confirmación explícita.
> Cada pantalla confirmada se enlaza con los IDs de función del Catálogo (03).
>
> PM Skills de eco: `storyboard`, `user-story-mapping`.
> La factoría mapea esto al contexto UI (futura gsd-ui-phase / UI-SPEC).
>
> Señal brownfield/greenfield:
>   - "lo que ya existe"  → señal brownfield para la factoría
>   - "lo que me gustaría" → señal greenfield

---

## Pantallas documentadas

*Una sección por cada diseño o captura analizada. Copia el bloque y rellénalo.*

---

### Pantalla P-001

**Nombre:** {{NOMBRE_PANTALLA_1}}
**Archivo de referencia:** `assets/{{ARCHIVO_1}}` *(ej. `assets/login-v2.png`)*
**¿Es un diseño existente o una idea de cómo debería verse?** Existente / Idea

**Propósito de la pantalla**
> *¿Para qué sirve? ¿En qué momento del recorrido aparece?*
> {{PROPOSITO_1}}

**Elementos detectados** *(solo lo que se ve — sin inventar)*
- {{ELEMENTO_1_A}} *(ej. campo de correo electrónico)*
- {{ELEMENTO_1_B}} *(ej. botón "Iniciar sesión")*
- {{ELEMENTO_1_C}} *(ej. enlace "¿Olvidaste tu contraseña?")*

**Acciones del usuario**
- {{ACCION_1_A}} *(ej. introduce correo y contraseña y pulsa el botón)*
- {{ACCION_1_B}}

**Funciones inferidas** *(confirmadas por el usuario — IDs del Catálogo 03)*
- {{F-001}} — {{DESCRIPCION_INFERIDA_1}}
- {{F-002}} — {{DESCRIPCION_INFERIDA_2}}

**Confirmado:** Sí / No / Ajustado
*Si "Ajustado", describe qué cambió:*
> {{AJUSTE_1}}

---

### Pantalla P-002

**Nombre:** {{NOMBRE_PANTALLA_2}}
**Archivo de referencia:** `assets/{{ARCHIVO_2}}`
**¿Es un diseño existente o una idea de cómo debería verse?** Existente / Idea

**Propósito de la pantalla**
> {{PROPOSITO_2}}

**Elementos detectados**
- {{ELEMENTO_2_A}}
- {{ELEMENTO_2_B}}

**Acciones del usuario**
- {{ACCION_2_A}}

**Funciones inferidas** *(IDs del Catálogo 03)*
- {{F-003}} — {{DESCRIPCION_INFERIDA_3}}

**Confirmado:** Sí / No / Ajustado

---

*Añade tantos bloques P-00X como diseños haya. Aura genera uno por cada archivo subido.*

---

## Pantallas que faltan

*Huecos detectados por Aura: pantallas que el recorrido implica pero que no se han diseñado todavía.
Anotadas como pendientes, no como bloqueantes.*

| Pantalla implícita | ¿Por qué se infiere? | ¿Necesaria para el MVP? |
|--------------------|----------------------|------------------------|
| {{PANTALLA_FALTANTE_1}} | {{RAZON_1}} | Sí / No / Por decidir |
| {{PANTALLA_FALTANTE_2}} | {{RAZON_2}} | Sí / No / Por decidir |

*Ejemplo:*
*`Pantalla de recuperación de contraseña | El login tiene enlace "¿Olvidaste tu contraseña?" | Por decidir`*
*`Pantalla de perfil del músico | La reserva implica ver el perfil antes | Sí`*
