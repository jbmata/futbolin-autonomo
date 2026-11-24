# Índice de la vault

Memoria de este proyecto, compartida por los agentes y por mí. Un archivo por hecho; enlaza con [[wikilinks]].

- [[Estructura]] — mapa del proyecto: dónde está cada cosa
- [[Estado]] — en qué punto está el proyecto ahora
- [[Pendientes]] — lo que falta por hacer
- `Decisiones/` — por qué se eligió X y no Y (un archivo por decisión)
- `Referencias/` — datasheets, enlaces, normativa, contactos
- `Diario/` — notas de sesión (AAAA-MM-DD.md)

## Decisiones
- [[2026-11-24-condensador-reset]] — restaurar C1033 de 100 nF en RESET
- [[2026-11-24-pcb-cuatro-capas-jlcpcb]] — máximo cuatro capas, menor área y proceso estándar de JLCPCB
- [[2026-11-24-medidas-fuente-unica]] — `Medidas/mesa.json` manda sobre el 3D y el simulador
- [[2026-11-24-topes-y-carrera-por-barra]] — pasadores, goma y muelle; cada barra con su carrera
- [[2026-11-24-referencia-desde-tope-motores]] — todo se mide desde el final de carrera de los motores
- [[2026-11-24-organizacion-carpeta-3d]] — `Mecanica/3d` en piezas, montajes, referencias, mesa y robot
- [[2026-11-23-compensacion-latencia]] — adelantar la bola y las barras rivales al instante actual
- [[2026-11-23-build123d-en-vez-de-openscad]] — piezas 3D en Python, un fichero por pieza
- [[2026-11-22-dagger-sin-ppo]] — imitación DAgger; PPO no se usa porque empeora al alumno

## Referencias
- [[hardware-mcu-rt]] — esquema, colocación de PCB, comprobaciones y límites antes de fabricar
- [[plano-barras-futbolinmadrid]] — largos de barra y agujeros de los jugadores
- [[datos-de-la-mesa-del-usuario]] — hechos de la mesa que dio el usuario
- [[historial-entrenamientos]] — runs de GitHub y resultados
- [[github-actions]] — workflows y ficheros de disparo
- [[lecciones-simulador]] — fallos no evidentes y cómo detectarlos
- [[motores-tema-aparcado]] — velocidades, polea y masas: no tocar sin permiso
- [[recorridos-artifact]] — vista interactiva de los recorridos
- [[montaje-con-motores]] — actuadores sobre la mesa: recorrido útil del robot y estado de los ficheros

## Diario
- [[2026-11-24]]
