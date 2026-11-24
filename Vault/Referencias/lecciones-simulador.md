# Lecciones del simulador

Fallos no evidentes encontrados (2026-11-21 a 2026-11-24). Repetir estas comprobaciones tras cualquier
cambio de física, del heurístico o de la mesa.

- Medir heurístico contra heurístico **con el agente en A y en B**: si gana siempre el mismo equipo hay una
  asimetría. Hubo dos: el orden de resolución de contactos por índice de barra y el heurístico multiplicando
  por la dirección del equipo un comando que ya estaba en el marco del mundo.
- La mesa debe ser **exactamente** simétrica por rotación de 180°: el agente juega con los dos equipos.
  Una diferencia de 2 mm entre las barras de cada equipo rompe las pruebas de simetría (2026-11-24).
- Un heurístico sin retardo de percepción defiende casi perfecto: se le da un retardo de 60–200 ms.
- Con el rival estático, portero y defensa tapan la portería: la fase 0 usa el rival «absent».
- Armar el golpe con la bola acercándose hace que el taco la arrastre hacia atrás (autogoles).
- Con la cabeza de la figura modelada, armar a 180° toca la bola: armar a unos −110°.
- Una barra ligera absorbe los tiros. Un motor paso a paso sin encoder pierde pasos al pisar la bola:
  por eso la placa lleva AS5600.
- Una sola semilla engaña: comparar con varias y con intervalo de confianza.
- Las pruebas que colocan la bola en y = 0 suponen un jugador en el centro: con mesas reales, poner la bola
  delante del jugador más cercano (2026-11-24).
- No editar el código mientras entrena: `SubprocVecEnv` recarga los módulos en cada fase.
- En el bash de esta máquina los heredocs de Python a veces se rompen (comillas, `\n`): escribir el script
  en un fichero y ejecutarlo.
