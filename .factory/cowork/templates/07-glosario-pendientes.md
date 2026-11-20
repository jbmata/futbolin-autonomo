# 07 · Glosario y pendientes — {{NOMBRE_PROYECTO}}

> Etapa 9 del intake (cierre). Aura recoge aquí el vocabulario propio del negocio,
> las suposiciones que quedan sin confirmar, las decisiones pendientes y las que ya se tomaron.
> Evita malentendidos entre lo que el usuario llama "X" y lo que el equipo técnico entiende por "X".
>
> PM Skill de eco: `prd-development`, `problem-statement`.
> La factoría mapea esto a `.factory/discovery.md` (vocabulario + suposiciones).

---

## Glosario

*Términos propios del negocio o del sector que tienen un significado específico en este proyecto.
Un glosario claro evita que la factoría construya la cosa equivocada.*

| Término | Significado en este proyecto |
|---------|------------------------------|
| {{TERMINO_1}} | {{SIGNIFICADO_1}} |
| {{TERMINO_2}} | {{SIGNIFICADO_2}} |
| {{TERMINO_3}} | *(añadir)* |

*Ejemplo de filas:*
*`Sesión | Una cita de entrenamiento entre un entrenador y un cliente (no confundir con sesión de usuario/login).`*
*`Plan | El conjunto de sesiones semanales asignadas a un cliente específico.`*
*`Completada | Estado de una sesión que ya ocurrió y fue confirmada por el entrenador.`*
*`Cliente | Usuario final que recibe el plan. Diferente de "entrenador" aunque ambos usan la app.`*

---

## Suposiciones

*Cosas que se dan por ciertas para poder avanzar, pero que nadie ha validado todavía.
Cada suposición tiene un riesgo implícito — la factoría las toma como verdaderas hasta que se diga lo contrario.*

| # | Suposición | Riesgo si es falsa | ¿Quién puede confirmarla? |
|---|------------|-------------------|--------------------------|
| S1 | {{SUPOSICION_1}} | {{RIESGO_1}} | {{QUIEN_1}} |
| S2 | {{SUPOSICION_2}} | {{RIESGO_2}} | {{QUIEN_2}} |
| S3 | *(añadir)* | | |

*Ejemplo:*
*`S1 | Los entrenadores tienen smartphone con acceso a internet durante las sesiones | Sin esto la app no tiene utilidad real | Preguntar a 3 entrenadores antes del lanzamiento`*
*`S2 | Los clientes no necesitan ver su historial en el MVP — solo el entrenador lo necesita | Si es falso, hay que añadir una pantalla de cliente que retrasaría el lanzamiento | Validar con 2 clientes actuales`*

---

## Pendientes por decidir

*Preguntas que quedaron abiertas durante el intake. Alguien tiene que responderlas
antes de que la factoría empiece a construir (o durante la fase de descubrimiento).*

- [ ] **{{PENDIENTE_1}}**
  *Contexto:* {{CONTEXTO_1}}
  *¿Quién decide?* {{QUIEN_DECIDE_1}}
  *¿Cuándo hay que decidirlo?* {{CUANDO_1}}

- [ ] **{{PENDIENTE_2}}**
  *Contexto:* {{CONTEXTO_2}}
  *¿Quién decide?* {{QUIEN_DECIDE_2}}
  *¿Cuándo hay que decidirlo?* {{CUANDO_2}}

*Ejemplo:*
*`[ ] ¿Los clientes pueden ver su propio plan sin que el entrenador les dé acceso?`*
*`Contexto: en la sesión no quedó claro si los clientes tienen login propio en el MVP.`*
*`¿Quién decide? El dueño del negocio.`*
*`¿Cuándo? Antes de que empiece la fase de construcción.`*

---

## Decisiones tomadas

*Decisiones que SÍ se tomaron durante el intake y que no deben reabrirse sin motivo.*
*La trazabilidad ayuda a no perder el contexto semanas después.*

| Decisión | Alternativas descartadas | Razón | Fecha |
|----------|--------------------------|-------|-------|
| {{DECISION_1}} | {{ALTERNATIVA_1}} | {{RAZON_1}} | {{FECHA_1}} |
| {{DECISION_2}} | {{ALTERNATIVA_2}} | {{RAZON_2}} | {{FECHA_2}} |

*Ejemplo:*
*`MVP solo web responsiva | App nativa iOS/Android | Tiempo y coste: la web responsiva cubre el 90% del caso de uso | {{FECHA}}`*
*`Sin integración de pagos en fase 1 | Stripe desde el principio | Queremos validar el uso antes de cobrar | {{FECHA}}`*
