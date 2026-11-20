# 03 · Catálogo de funciones — {{NOMBRE_PROYECTO}}

> Etapas 3, 4 y 6 del intake. La tabla central es el corazón del brief: captura TODAS las
> funciones (dicho / diseño / inferido), con su prioridad MoSCoW y si entran en el MVP.
> Nunca borrar — etiquetar. Los "Won't-now" son la visión completa, fase 2+.
>
> PM Skills de eco: `user-story`, `user-story-mapping`, `prioritization-advisor`.
> La factoría mapea esto al roadmap (fases gsd / specs de spec-kit).
>
> Regla de origen:
>   - `dicho`    → el usuario lo mencionó en conversación
>   - `diseño`   → se extrajo de una captura / diseño subido (ver 04)
>   - `inferido` → Aura lo dedujo del contexto y el usuario lo confirmó

---

## Tabla de funciones

*Añade una fila por función. IDs en formato F-001, F-002, … para que 04 pueda referenciarlos.*

| ID | Función (lenguaje claro, sin jerga) | Usuario | Etapa del viaje | Prioridad MoSCoW | MVP? | Origen |
|----|--------------------------------------|---------|-----------------|------------------|------|--------|
| F-001 | {{FUNCION_1}} | {{USUARIO_1}} | {{ETAPA_1}} | Must | Sí | dicho |
| F-002 | {{FUNCION_2}} | {{USUARIO_2}} | {{ETAPA_2}} | Must | Sí | diseño |
| F-003 | {{FUNCION_3}} | {{USUARIO_3}} | {{ETAPA_3}} | Should | No | inferido |
| F-004 | *(añadir)* | | | | | |

*Ejemplo de fila:*
*`F-001 | Iniciar sesión con correo y contraseña | Entrenador | Acceso | Must | Sí | dicho`*
*`F-005 | Exportar historial de sesiones a PDF | Entrenador | Reportes | Could | No | inferido`*

**Prioridad MoSCoW:**
- **Must** — sin esto no hay MVP
- **Should** — muy deseable, intentar meter en MVP
- **Could** — buena idea, queda fuera del MVP
- **Won't-now** — visión completa, fase 2+

---

## Cara al usuario

*Funciones que el usuario final ve y toca directamente. Sección de resumen narrativo
(la tabla arriba es la referencia canónica).*

- {{CARA_USUARIO_1}}
- {{CARA_USUARIO_2}}

*Ejemplo:*
- *Login / registro de entrenadores y clientes.*
- *Ver el plan semanal asignado por el entrenador.*
- *Marcar una sesión como completada.*

---

## Backoffice / administración

*Funciones de gestión interna: configuración, administración de usuarios, revisión de datos.
Normalmente para un perfil admin o el propio negocio.*

- {{BACKOFFICE_1}}
- {{BACKOFFICE_2}}

*Ejemplo:*
- *Panel de administrador para ver todos los entrenadores y clientes activos.*
- *Desactivar una cuenta sin borrar los datos históricos.*

---

## Automatizaciones

*Cosas que ocurren solas, sin que nadie haga clic: notificaciones, recordatorios, sincronizaciones,
cálculos periódicos, integraciones que corren en segundo plano.*

- {{AUTOMATIZACION_1}} — *se dispara cuando:* {{TRIGGER_1}}
- {{AUTOMATIZACION_2}} — *se dispara cuando:* {{TRIGGER_2}}

*Ejemplo:*
- *Recordatorio por correo/push 1 hora antes de una sesión — se dispara con cada sesión programada.*
- *Actualizar el contador de sesiones del cliente al marcar como completada.*

---

## Tareas periódicas

*Procesos que corren en un horario fijo: resúmenes semanales, purgas de datos, informes mensuales.*

- {{PERIODICA_1}} — *frecuencia:* {{FRECUENCIA_1}}

*Ejemplo:*
- *Resumen semanal de sesiones completadas enviado por correo al entrenador — cada lunes a las 8:00.*

---

## Fuera de alcance (explícito)

*Cosas que el usuario mencionó o que Aura infirió pero que quedan fuera del MVP y de la fase 1.
Listadas aquí para evitar malentendidos con la factoría.*

- {{FUERA_1}}
- {{FUERA_2}}

*Ejemplo:*
- *App nativa iOS/Android — el MVP es web responsiva únicamente.*
- *Integración con plataformas de pago — los cobros siguen siendo manuales en fase 1.*
- *Multiidioma — solo español en fase 1.*
