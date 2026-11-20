<!-- Managed by the Software Factory. Formato: .factory/assets/templates/PENDIENTES.md (fuente: factory/templates/). -->
# PENDIENTES

Dudas, decisiones y aprobaciones que la factoría deja para una persona mientras trabaja sola
(`mode: ultra-autonomous` en `.factory/config.yml`). La factoría nunca se para a esperar una respuesta: anota aquí
lo que necesita, toma el supuesto seguro o aparca lo que no puede hacer sin ti, y sigue con el resto.

**Cómo responder.** Escribe tu respuesta en `Respuesta:` y cambia `Estado:` a `resuelta`, o díselo al agente en el
chat y él lo anota. `factory-build` lee este fichero al empezar cada ejecución y cada fase: aplica las respuestas,
desaparca las tareas que dependían de ellas, deshace los supuestos que rechaces y replanifica la fase afectada.

**Reglas del fichero.**

- Una entrada por asunto, con id `P-001`, `P-002`… Las entradas nuevas se **añaden al final** (append-only, ids
  crecientes). Una entrada no se borra: se marca `resuelta`.
- `Tipo`: `decisión` (la factoría eligió un default reversible y puedes cambiarlo), `duda` (falta información),
  `aprobación` (algo que la factoría nunca hace sola: merge o push a la rama principal, desplegar, secretos o
  variables de GitHub, borrar datos/ramas/repos/recursos cloud, gastar dinero, escribir a personas) o `bloqueo` (un
  gate falla igual 3 veces o falta algo que solo una persona puede hacer).
- `Mientras tanto`: `continuó` (siguió con el supuesto), `aparcó la tarea` o `aparcó la fase` (y lo que dependa).
- `Estado`: `abierta` o `resuelta`.
- Se commitea con el trabajo: es estado del proyecto. **Nunca escribas secretos** (tokens, contraseñas, claves):
  di dónde están (`.env`, `secrets.env`) o qué hay que crear, nunca su valor.
- La sección `Última ejecución` la reescribe `factory-build` al terminar cada ejecución; no es una entrada.

## Última ejecución

<!-- factory-build: fecha · fases hechas N · aparcadas N · bloqueadas N · pendientes abiertas N -->

<!-- Copia este bloque al final del fichero por cada entrada nueva (fuera del comentario):

## P-001 · Título breve

- **Fecha:** AAAA-MM-DD
- **Fase/tarea:** phase-N / tarea
- **Tipo:** decisión
- **Contexto:** qué estaba haciendo la factoría y por qué necesita a una persona.
- **Opciones:** A) … · B) …
- **Supuesto de la factoría:** qué asumió (reversible: sí | no), o «ninguno» si aparcó.
- **Mientras tanto:** continuó
- **Estado:** abierta
- **Respuesta:**

La factoría añade `- **Aplicada:** AAAA-MM-DD` cuando ya aplicó la respuesta; una entrada resuelta sin esa línea se
aplica en la siguiente ejecución o fase.
-->
