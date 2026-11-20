# 06 · Restricciones e integraciones — {{NOMBRE_PROYECTO}}

> Etapa 8 del intake. Aura aplaza intencionalmente este bloque hasta el final
> para no estresar al usuario al principio. Recoge lo que la plataforma tiene que
> conectar, dónde tiene que funcionar y qué reglas no se pueden saltarse.
>
> PM Skill de eco: `prd-development`.
> La factoría mapea esto a `.factory/integrations/` y a la elección de starterkit/stack.
>
> Nota sobre secretos: NUNCA escribir aquí contraseñas, tokens, API keys ni credenciales.
> Solo nombres de sistemas y variables. Los secretos van en archivos `.env` que no se suben al repo.

---

## Integraciones obligatorias

*Sistemas externos con los que la plataforma TIENE que hablar para funcionar en el MVP.*

| Sistema / Servicio | Para qué se necesita | ¿Ya existe una cuenta/contrato? | Prioridad |
|--------------------|----------------------|---------------------------------|-----------|
| {{SISTEMA_1}} | {{PARA_QUE_1}} | Sí / No / No sé | Must / Should |
| {{SISTEMA_2}} | {{PARA_QUE_2}} | Sí / No / No sé | Must / Should |
| {{SISTEMA_3}} | *(añadir si hace falta)* | | |

*Ejemplo de filas:*
*`Stripe | Cobrar suscripciones mensuales | Sí, tenemos cuenta | Must`*
*`SendGrid | Enviar correos transaccionales (confirmaciones, recordatorios) | No | Should`*
*`Google Calendar | Sincronizar sesiones con el calendario del entrenador | No sé | Could`*

*Si no hay integraciones obligatorias para el MVP, escribe: "Ninguna — el MVP funciona de forma autónoma."*

---

## Plataformas y dispositivos

*¿Dónde tiene que funcionar la plataforma? Marca todo lo que aplique.*

| Plataforma | MVP | Fase 2+ | Notas |
|------------|-----|---------|-------|
| Web (escritorio, Chrome/Firefox/Safari) | ☐ | ☐ | |
| Web responsiva (móvil desde navegador) | ☐ | ☐ | |
| App nativa iOS | ☐ | ☐ | |
| App nativa Android | ☐ | ☐ | |
| Tablet | ☐ | ☐ | |
| Impresión / PDF generado | ☐ | ☐ | |
| {{OTRA_PLATAFORMA}} | ☐ | ☐ | |

---

## Idiomas

*¿En qué idiomas tiene que estar disponible la plataforma?*

| Idioma | MVP | Fase 2+ |
|--------|-----|---------|
| Español | ☐ | ☐ |
| Inglés | ☐ | ☐ |
| {{OTRO_IDIOMA}} | ☐ | ☐ |

*Si solo hay un idioma en el MVP, escribe aquí: "Solo {{IDIOMA}} en fase 1 — multiidioma diferido."*

---

## Reglas legales, privacidad y accesos

*¿Hay regulaciones, restricciones de datos o requisitos de acceso que la plataforma tiene que cumplir?
Descrito en lenguaje de negocio — el equipo técnico adaptará.*

- **Privacidad de datos:** {{PRIVACIDAD}}
  *(ej. "Manejamos datos de salud — necesitamos cumplir con RGPD y posiblemente HIPAA")*
- **Menores de edad:** {{MENORES}}
  *(ej. "Los usuarios pueden ser menores — necesitamos consentimiento de los padres")*
- **Acceso geográfico:** {{ACCESO_GEO}}
  *(ej. "Solo usuarios de España en fase 1" o "Sin restricciones geográficas")*
- **Requisitos de facturación:** {{FACTURACION}}
  *(ej. "Necesitamos emitir facturas con IVA — integración con herramienta contable")*
- **Otros requisitos legales:** {{OTROS_LEGAL}}

---

## Sistemas y datos existentes

*¿Hay información o sistemas que ya existen y con los que la nueva plataforma tiene que convivir?*

| Sistema / dato existente | Qué contiene | ¿Hay que migrar datos? | Prioridad |
|--------------------------|--------------|------------------------|-----------|
| {{SISTEMA_EXISTENTE_1}} | {{CONTIENE_1}} | Sí / No / Parcial | Must / Should / Could |
| {{SISTEMA_EXISTENTE_2}} | {{CONTIENE_2}} | Sí / No / Parcial | |

*Ejemplo:*
*`Hoja de cálculo Google Sheets | 200 clientes con sus datos de contacto | Sí, importación inicial | Should`*
*`Sistema de reservas antiguo | Historial de sesiones 2022-2024 | No — histórico se archiva | Could`*

*Si no hay sistemas existentes: "Proyecto greenfield — sin datos ni sistemas que migrar."*

---

## Suposiciones técnicas a validar

*Cosas que se asumen como ciertas para arrancar, pero que alguien tiene que confirmar
antes o durante la construcción. Aura las marca para que no se pierdan.*

- [ ] {{SUPOSICION_TECNICA_1}}
- [ ] {{SUPOSICION_TECNICA_2}}

*Ejemplo:*
- *[ ] Asumimos que SendGrid tiene plan gratuito suficiente para el MVP (< 100 correos/día).*
- *[ ] Asumimos que los entrenadores tienen acceso estable a internet desde el gimnasio.*
- *[ ] Asumimos que podemos usar Auth0 en el plan gratuito para el MVP (< 7.000 MAU).*
