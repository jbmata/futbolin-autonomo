# 02 · Usuarios objetivo — {{NOMBRE_PROYECTO}}

> Etapa 2 del intake. Aura identifica los tipos de persona que usarán la plataforma,
> construye una proto-persona por tipo y captura el trabajo principal que cada una quiere lograr
> (JTBD). También define quién puede ver qué (visibilidad/permisos) y el volumen esperado.
>
> PM Skills de eco: `proto-persona`, `jobs-to-be-done`.
> La factoría mapea esto a `.factory/discovery.md` (personas, JTBD, roles/permisos).

---

## Tipos de usuario

*Lista de los distintos perfiles de persona que usarán la plataforma. No roles técnicos —
personas reales con nombres descriptivos.*

| # | Nombre del tipo | Descripción en una frase | ¿Tiene acceso al MVP? |
|---|-----------------|--------------------------|----------------------|
| U1 | {{TIPO_1}} | {{DESC_1}} | Sí / No / Parcial |
| U2 | {{TIPO_2}} | {{DESC_2}} | Sí / No / Parcial |
| U3 | *(añadir si hace falta)* | | |

*Ejemplo de fila: `U1 | Entrenador personal | El que crea planes y sigue a clientes | Sí`*

---

## Proto-persona por tipo

*Una proto-persona breve por cada tipo de usuario del MVP. Aura las redacta y el usuario las corrige.*

### U1 · {{TIPO_1}}

- **Nombre ficticio:** {{NOMBRE_FICTICIO_1}}
- **Situación:** {{SITUACION_1}}
  *(ej. "Lleva 5 años como entrenador autónomo, 30 clientes activos, usa WhatsApp y Google Sheets")*
- **Frustración principal:** {{FRUSTRACION_1}}
- **Lo que más valora:** {{VALORA_1}}
- **Cómo usaría la plataforma:** {{USO_1}}

### U2 · {{TIPO_2}}

- **Nombre ficticio:** {{NOMBRE_FICTICIO_2}}
- **Situación:** {{SITUACION_2}}
- **Frustración principal:** {{FRUSTRACION_2}}
- **Lo que más valora:** {{VALORA_2}}
- **Cómo usaría la plataforma:** {{USO_2}}

---

## Trabajo principal — Jobs to Be Done (JTBD)

*Para cada tipo de usuario: el trabajo que quiere lograr, en su propio idioma.
Formato: "Cuando [situación], quiero [motivación], para [resultado deseado]."*

| Usuario | Trabajo principal (JTBD) |
|---------|--------------------------|
| U1 · {{TIPO_1}} | Cuando {{SITUACION_JTBD_1}}, quiero {{MOTIVACION_1}}, para {{RESULTADO_1}}. |
| U2 · {{TIPO_2}} | Cuando {{SITUACION_JTBD_2}}, quiero {{MOTIVACION_2}}, para {{RESULTADO_2}}. |

*Ejemplo: `U1 · Entrenador | Cuando empiezo la semana, quiero ver de un vistazo qué sesiones tengo,
para no tener que abrir cinco conversaciones de WhatsApp.`*

---

## Visibilidad y permisos

*¿Quién puede ver qué? Descrito en lenguaje de negocio, no de base de datos.*
*La factoría traduce esto a roles/permisos en el modelo de datos.*

| Acción o dato | {{TIPO_1}} | {{TIPO_2}} | Admin / Backoffice |
|---------------|------------|------------|-------------------|
| Ver su propio perfil | Sí | Sí | Sí |
| Ver perfil de otros usuarios | {{PERMISO_11}} | {{PERMISO_21}} | Sí |
| Crear / editar contenido propio | {{PERMISO_12}} | {{PERMISO_22}} | Sí |
| Ver contenido de otros | {{PERMISO_13}} | {{PERMISO_23}} | Sí |
| Acciones de administración | No | No | Sí |

*Ajusta las filas según lo que emerja en el intake.*

---

## Volumen esperado

*Estimaciones a ojo — para dimensionar la arquitectura, no para un business plan.*

| Métrica | Estimación inicial |
|---------|--------------------|
| Usuarios totales al lanzar | {{USUARIOS_LANZAMIENTO}} |
| Usuarios activos al mes (mes 3) | {{MAU_MES3}} |
| Pico de uso simultáneo | {{PICO}} |
| Volumen de datos sensibles | {{DATOS_SENSIBLES}} *(ej. "fotos de perfil, nada médico")* |

*Si no se sabe, escribe "no lo sé todavía" — la factoría usará defaults conservadores.*
