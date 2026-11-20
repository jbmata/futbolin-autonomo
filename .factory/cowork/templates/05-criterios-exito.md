# 05 · Criterios de éxito — {{NOMBRE_PROYECTO}}

> Etapa 7 del intake. Esto es el **contrato de parada** del bucle de construcción:
> la factoría construye hasta que todos los criterios pueden marcarse como cumplidos con evidencia.
> Aura insiste en que sean MEDIBLES — rechaza los vagos ("funciona bien", "es rápido").
>
> PM Skills de eco: `press-release`, `prd-development`.
> La factoría mapea esto a `FACTORY.md` (success criteria / stop condition del bucle).
>
> Regla de oro: si no puedes escribir la prueba antes de construir, el criterio es demasiado vago.

---

## Definición de "funciona" — día 1

*¿Qué tiene que pasar para que el día del lanzamiento digas "esto funciona"?
Descrito como lo vería alguien de fuera, sin ver el código.*

> {{DEFINICION_DIA_1}}
>
> *Ejemplo: "Un entrenador puede registrarse, crear un cliente, asignarle un plan semanal
> y marcar una sesión como completada — todo desde el móvil, en menos de 5 minutos,
> sin que le expliquen nada."*

---

## Métricas al mes — MEDIBLES

*2 a 4 resultados concretos y verificables que quieres haber alcanzado al cabo de un mes de uso real.
Nada de "muchos usuarios" ni "buena experiencia" — números o hechos observables.*

| # | Métrica | Objetivo al mes | Cómo se mide |
|---|---------|-----------------|--------------|
| M1 | {{METRICA_1}} | {{OBJETIVO_1}} | {{COMO_MEDIR_1}} |
| M2 | {{METRICA_2}} | {{OBJETIVO_2}} | {{COMO_MEDIR_2}} |
| M3 | *(opcional)* | | |

*Ejemplo de filas:*
*`M1 | Entrenadores activos (han creado ≥1 plan) | 10 | contador en base de datos`*
*`M2 | Sesiones marcadas como completadas | ≥50 en el mes | log de la acción`*
*`M3 | Tiempo medio en registrarse y crear el primer plan | < 5 min | timestamps de eventos`*

---

## Señales de que valió la pena

*¿Qué verías o escucharías que te haría pensar "menos mal que lo hicimos"?
Más cualitativo — comportamiento de los usuarios o feedback espontáneo.*

- {{SENAL_1}}
- {{SENAL_2}}
- {{SENAL_3}}

*Ejemplo:*
- *Los entrenadores abren la app antes de la primera sesión del día sin que nadie se lo diga.*
- *Algún entrenador recomienda la herramienta a otro sin que lo hayamos pedido.*
- *Ningún entrenador pregunta "¿cómo hago X?" en el grupo de soporte.*

---

## Criterios de aceptación del MVP

*Lista técnica de lo que debe poder hacerse antes de salir — el checklist que el equipo usa
para dar por terminado el MVP. Aura la redacta, el usuario la valida.*

*Regla: cada criterio tiene que ser verificable por un humano o por un test automatizado.*

- [ ] {{CA_1}}
- [ ] {{CA_2}}
- [ ] {{CA_3}}
- [ ] {{CA_4}}

*Ejemplo:*
- *[ ] Un nuevo usuario puede registrarse con correo y contraseña y recibe un correo de confirmación.*
- *[ ] Un entrenador puede crear un plan semanal con al menos 5 sesiones y asignarlo a un cliente.*
- *[ ] Un cliente puede ver su plan desde el móvil sin necesitar ordenador.*
- *[ ] Marcar una sesión como completada actualiza el contador del cliente en tiempo real.*
- *[ ] La app carga en menos de 3 segundos en una conexión 4G normal.*

---

## No-goals explícitos

*Cosas que podrían parecer parte del MVP pero que conscientemente dejamos fuera.
Fundamental para que la factoría sepa cuándo parar.*

- {{NO_GOAL_1}}
- {{NO_GOAL_2}}

*Ejemplo:*
- *No se miden resultados físicos ni se integra con wearables en fase 1.*
- *No hay app nativa — el MVP es web responsiva únicamente.*
- *No hay sistema de pagos integrado — los cobros son externos al MVP.*
