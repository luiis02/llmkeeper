# Referencia de la API

Base URL: `http://127.0.0.1:8000/api/v1`

Documentación interactiva (generada por FastAPI): `http://127.0.0.1:8000/docs` (Swagger UI) y `/redoc`.

Todas las peticiones y respuestas usan JSON (`Content-Type: application/json`). No hay autenticación.

---

## Resumen

| Método | Ruta | Descripción |
|--------|------|-------------|
| `POST` | [`/fields`](#post-fields) | Crear un campo temático |
| `GET` | [`/fields/{field_id}`](#get-fieldsfield_id) | Consultar un campo |
| `PATCH` | [`/fields/{field_id}/strictness`](#patch-fieldsfield_idstrictness) | Cambiar la estrictud |
| `POST` | [`/fields/{field_id}/validate`](#post-fieldsfield_idvalidate) | Validar un mensaje (sin estado) |
| `POST` | [`/fields/{field_id}/sessions`](#post-fieldsfield_idsessions) | Crear una sesión |
| `GET` | [`/sessions/{session_id}`](#get-sessionssession_id) | Consultar una sesión |
| `POST` | [`/sessions/{session_id}/messages`](#post-sessionssession_idmessages) | Enviar un mensaje a una sesión |

---

## Tipos comunes

### `StrictnessLevel` (string)

`muy_estricto` · `estricto` · `moderado` · `permisivo` · `muy_permisivo`

Umbrales asociados: 0.32 · 0.27 · 0.22 · 0.17 · 0.12. Ver [puntuacion.md](puntuacion.md#niveles-de-estrictud).

### `SessionStatus` (string)

`on_topic` · `warning` · `off_topic`. Ver [flujos.md](flujos.md#flujo-de-sesión-y-estados).

### Errores

```json
{ "detail": "Field not found" }
```

`404` con `Field not found` o `Session not found`. `422` con la lista de errores de validación de Pydantic.

---

## `POST /fields`

Crea un campo temático a partir de un *system prompt*.

### Cuerpo

| Campo | Tipo | Obligatorio | Por defecto | Descripción |
|-------|------|-------------|-------------|-------------|
| `system_prompt` | string | sí | — | Texto que define el tema permitido |
| `extra_topics` | string[] | no | `null` | Temas adicionales que se añaden como anclas |
| `strictness_preset` | `StrictnessLevel` | no | `moderado` | Nivel de estrictud |
| `strictness_value` | number, `[0, 1]` | no | `null` | Umbral personalizado. Si se envía, tiene prioridad sobre el preset |

```bash
curl -X POST http://127.0.0.1:8000/api/v1/fields \
  -H "Content-Type: application/json" \
  -d '{
    "system_prompt": "- Recetas de cocina\n- Ingredientes y sustituciones\n- Técnicas culinarias",
    "extra_topics": ["Maridaje de vinos con comida"],
    "strictness_preset": "estricto"
  }'
```

### Respuesta `200`

```json
{
  "field_id": "e2d8f5b6-7c1a-4f0e-9d3b-2a6c8e1f4b90",
  "system_prompt": "- Recetas de cocina\n- Ingredientes y sustituciones\n- Técnicas culinarias",
  "num_anchors": 4,
  "dispersion": 0.29,
  "threshold": 0.27,
  "preset": "estricto",
  "alpha": 0.7,
  "top_k": 3
}
```

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `field_id` | string (UUID) | Identificador del campo |
| `system_prompt` | string | Eco del prompt recibido |
| `num_anchors` | int | Anclas totales (prompt + extra) |
| `dispersion` | float | Distancia coseno media al centroide |
| `threshold` | float | Umbral efectivo |
| `preset` | string \| null | Preset usado, o `null` si el umbral es personalizado |
| `alpha` | float | Peso de las anclas frente al centroide (fijo en 0.7) |
| `top_k` | int | Número de anclas promediadas (fijo en 3) |

### Errores

| Código | Caso |
|--------|------|
| `422` | `strictness_value` fuera de `[0, 1]`, preset desconocido o cuerpo inválido |

---

## `GET /fields/{field_id}`

Devuelve los metadatos de un campo. Mismo formato que la respuesta de `POST /fields`.

```bash
curl http://127.0.0.1:8000/api/v1/fields/e2d8f5b6-7c1a-4f0e-9d3b-2a6c8e1f4b90
```

| Código | Caso |
|--------|------|
| `404` | Campo inexistente (`Field not found`) |

---

## `PATCH /fields/{field_id}/strictness`

Cambia el umbral de un campo.

### Cuerpo

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `preset` | `StrictnessLevel` \| null | Preset a aplicar |
| `custom_value` | number `[0, 1]` \| null | Umbral personalizado. Tiene prioridad sobre `preset` |

Si ambos son `null` u omitidos, el campo vuelve a `moderado`.

```bash
curl -X PATCH http://127.0.0.1:8000/api/v1/fields/<field_id>/strictness \
  -H "Content-Type: application/json" \
  -d '{"preset": "muy_estricto"}'
```

### Respuesta `200`

```json
{ "message": "Strictness updated" }
```

Para ver el nuevo umbral, consulta `GET /fields/{field_id}`.

| Código | Caso |
|--------|------|
| `404` | Campo inexistente |
| `422` | `custom_value` fuera de `[0, 1]` |

---

## `POST /fields/{field_id}/validate`

Puntúa un mensaje contra el campo **sin guardar nada**.

### Cuerpo

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `text` | string | Mensaje a evaluar |

```bash
curl -X POST http://127.0.0.1:8000/api/v1/fields/<field_id>/validate \
  -H "Content-Type: application/json" \
  -d '{"text": "¿Qué ingredientes necesito para una paella?"}'
```

### Respuesta `200`

```json
{
  "is_valid": true,
  "score": 0.64,
  "threshold": 0.22,
  "nearest_anchor_text": "Recetas de cocina"
}
```

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `is_valid` | bool | `score >= threshold` |
| `score` | float | Puntuación híbrida ([fórmula](puntuacion.md#fórmula-de-puntuación)) |
| `threshold` | float | Umbral aplicado |
| `nearest_anchor_text` | string | Ancla con mayor similitud coseno |

| Código | Caso |
|--------|------|
| `404` | Campo inexistente |
| `422` | Falta `text` o no es string |

---

## `POST /fields/{field_id}/sessions`

Crea una sesión de conversación asociada a un campo.

### Cuerpo

| Campo | Tipo | Por defecto | Descripción |
|-------|------|-------------|-------------|
| `patience` | int | `3` | Mensajes fuera de tema consecutivos para pasar a `off_topic` |
| `rolling_window` | int | `5` | Tamaño de la ventana de la media móvil |

```bash
curl -X POST http://127.0.0.1:8000/api/v1/fields/<field_id>/sessions \
  -H "Content-Type: application/json" \
  -d '{"patience": 3, "rolling_window": 5}'
```

### Respuesta `200`

```json
{
  "session_id": "6f1c2a9e-3b4d-4e8a-8c7f-0d9e2b1a5c33",
  "field_id": "e2d8f5b6-7c1a-4f0e-9d3b-2a6c8e1f4b90",
  "patience": 3,
  "rolling_window": 5,
  "status": "on_topic",
  "rolling_average": 1.0,
  "num_messages": 0
}
```

> `rolling_average` vale `1.0` cuando no hay mensajes; no indica nada todavía.

| Código | Caso |
|--------|------|
| `404` | Campo inexistente |

---

## `GET /sessions/{session_id}`

Devuelve el estado de una sesión. No incluye el historial de mensajes.

```bash
curl http://127.0.0.1:8000/api/v1/sessions/<session_id>
```

### Respuesta `200`

```json
{
  "session_id": "6f1c2a9e-3b4d-4e8a-8c7f-0d9e2b1a5c33",
  "field_id": "e2d8f5b6-7c1a-4f0e-9d3b-2a6c8e1f4b90",
  "patience": 3,
  "rolling_window": 5,
  "status": "warning",
  "rolling_average": 0.41,
  "num_messages": 4
}
```

| Código | Caso |
|--------|------|
| `404` | Sesión inexistente (`Session not found`) |

> La respuesta no incluye `consecutive_off`. Si lo necesitas, cuéntalo a partir de los resultados de `POST /messages`.

---

## `POST /sessions/{session_id}/messages`

Añade un mensaje al historial de la sesión, lo puntúa y actualiza el estado.

**Efecto:** modifica la sesión, aunque el mensaje sea inválido.

### Cuerpo

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `text` | string | Mensaje del usuario |

```bash
curl -X POST http://127.0.0.1:8000/api/v1/sessions/<session_id>/messages \
  -H "Content-Type: application/json" \
  -d '{"text": "¿Cuál es la capital de Francia?"}'
```

### Respuesta `200`

```json
{
  "score": 0.18,
  "is_valid": false,
  "threshold": 0.22,
  "rolling_average": 0.41,
  "status": "warning",
  "consecutive_off": 1
}
```

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `score` | float | Puntuación de **este** mensaje |
| `is_valid` | bool | Si este mensaje está dentro de tema |
| `threshold` | float | Umbral con el que se evaluó este mensaje |
| `rolling_average` | float | Media de los últimos `rolling_window` scores |
| `status` | string | Estado de la sesión **tras** este mensaje |
| `consecutive_off` | int | Mensajes fuera de tema consecutivos |

| Código | Caso |
|--------|------|
| `404` | Sesión inexistente (`Session not found`) |
| `422` | Falta `text` |

---

## Ejemplo de integración

Flujo mínimo en Python con `httpx`, que decide si reenviar cada mensaje al LLM:

```python
import httpx

BASE = "http://127.0.0.1:8000/api/v1"
client = httpx.Client(base_url=BASE)

field = client.post("/fields", json={
    "system_prompt": "Eres un experto en cocina. Ayudas con recetas e ingredientes.",
    "strictness_preset": "moderado",
}).json()

session = client.post(
    f"/fields/{field['field_id']}/sessions",
    json={"patience": 2},
).json()


def procesar(texto: str) -> str:
    r = client.post(f"/sessions/{session['session_id']}/messages", json={"text": texto}).json()
    if r["status"] == "off_topic":
        return "Esa pregunta está fuera del tema de esta conversación."
    # Aquí llamarías al LLM real
    return f"[enviado al LLM] {texto}"
```
