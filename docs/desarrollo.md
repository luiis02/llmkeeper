# Desarrollo

Guía para trabajar en el proyecto: entorno, tests, convenciones para extenderlo y limitaciones conocidas.

## Entorno

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Dependencias de runtime (`pyproject.toml`): `fastapi`, `uvicorn[standard]`, `sentence-transformers`, `torch`, `numpy`, `pydantic`, `pydantic-settings`.

Dependencias de desarrollo: `pytest`, `pytest-asyncio`, `httpx`.

## Tests

### Suite rápida (por defecto)

```bash
pytest
```

- Excluye los tests marcados como `integration` (`addopts` en `pyproject.toml`).
- No carga el modelo real. Usa `tests/fakes.py` para embeddings deterministas.
- Tarda ~1–2 s. Estado actual: **30 tests pasando**.

### Tests de integración

```bash
pytest -m integration
```

- Cargan `paraphrase-multilingual-MiniLM-L12-v2` real.
- Son lentos y necesitan descargar el modelo la primera vez.
- Validan la semántica en español con ejemplos reales.

### Qué prueba cada carpeta

| Carpeta | Qué cubre | Tipo |
|---------|-----------|------|
| `tests/domain/` | Chunking, centroide, dispersión, puntuación y transiciones de sesión | Unitarios puros |
| `tests/application/` | Casos de uso con repositorios en memoria y embeddings falsos | Unitarios con dobles |
| `tests/adapters/test_api.py` | Rutas HTTP, códigos de error y validación, con un `FastAPI()` mínimo y embeddings falsos | Integración de la API sin modelo |
| `tests/adapters/test_api_integration.py` | Flujo HTTP completo con `create_app` y el modelo real | Integración (`integration`) |
| `tests/adapters/test_prompt_injection_integration.py` | Validación con el modelo real de prompts en español (fuera y dentro de tema) | Integración (`integration`) |

### Dobles de prueba

`tests/fakes.py` contiene dos proveedores de embeddings:

- **`FakeEmbeddingProvider`**: hash SHA-256 del texto. Determinista, pero sin significado semántico. Sirve para tests que solo necesitan vectores estables.
- **`ScriptedEmbeddingProvider`**: devuelve vectores de una tabla fija y lanza `KeyError` si el texto no está en ella. Permite construir escenarios exactos:

```python
embeddings = ScriptedEmbeddingProvider({
    "Recetas de cocina": [1.0, 0.0, 0.0],
    "¿Cómo hago una paella?": [0.9, 0.1, 0.0],
    "¿Capital de Francia?": [0.0, 0.0, 1.0],
})
```

Con vectores construidos a mano puedes comprobar la fórmula, los umbrales y las transiciones de estado sin depender del modelo. Si un test falla con `KeyError`, casi siempre es porque el texto no está en el mapa.

### Escribir un test nuevo

- Lógica de dominio: prueba la función pura directamente, sin fixtures.
- Caso de uso: usa los repositorios en memoria y `ScriptedEmbeddingProvider`.
- Endpoint sin modelo: monta un `FastAPI()` con `create_router(...)` y un `Container` con `ScriptedEmbeddingProvider`, como hace `test_api.py`.
- Endpoint con la app completa: `create_app(Settings())` carga el modelo real en el `lifespan`. Usa `with TestClient(app) as client:`; sin el bloque `with` el `lifespan` no se ejecuta y las rutas no están registradas.

## Cómo extender el proyecto

### Añadir un nuevo proveedor de embeddings

1. Crea `src/llmkeeper/adapters/outbound/<nombre>.py`.
2. Implementa `embed(text) -> Embedding` y `embed_batch(texts) -> list[Embedding]`. Ambos deben devolver `Embedding` (ver `domain/models.py`).
3. En `infrastructure/app.py`, sustituye `SentenceTransformerEmbeddingAdapter` por tu clase, o hazlo configurable con `Settings`.
4. No hace falta tocar dominio ni casos de uso.

### Añadir un repositorio persistente (SQLite)

1. Crea `adapters/outbound/repository_sqlite.py` con clases que cumplan `FieldRepositoryPort` y `SessionRepositoryPort`.
2. Serializa `TopicField` y `Session`. Ten en cuenta que las anclas guardan vectores `float32`, y que `Session.history` es una lista de `MessageScore`.
3. Cambia `Container` para usar los repositorios SQLite y pasa `settings.db_path`.
4. Los tests de casos de uso deberían seguir funcionando sin cambios, porque dependen de los puertos.

### Añadir un endpoint

1. Si necesita lógica nueva, crea primero el caso de uso en `application/use_cases.py`.
2. Añade los esquemas en `adapters/inbound/schemas.py`.
3. Registra la ruta en `adapters/inbound/api.py`. Traduce `FieldNotFoundError` y `SessionNotFoundError` a `404`.
4. Escribe el test en `tests/adapters/test_api.py`.

### Cambiar la puntuación

La fórmula está en `drift_evaluator.score_against_field`. Si cambias `alpha` o `top_k`, recuerda que hoy están fijados en `TopicField` (`alpha: float = 0.7`, `top_k: int = 3`) y que `build_topic_field` usa esos valores por defecto. Los tests de `test_drift_evaluator.py` comprueban la fórmula con vectores conocidos; actualízalos junto con la fórmula.

## Scripts de investigación

`scripts/injection_direction.py` es un experimento independiente para detectar *prompt injections* con una dirección vectorial. No forma parte del servicio y tiene dependencias propias (`datasets`, `pulp`, `chromadb`) que no están en `pyproject.toml`. Ver [scripts/README.md](../scripts/README.md).

## Limitaciones conocidas

| Área | Limitación | Impacto |
|------|------------|---------|
| Persistencia | Repositorios en memoria | Se pierden campos y sesiones al reiniciar |
| Configuración | `DB_PATH` y `DEFAULT_*` no se leen | Cambiarlos no tiene efecto |
| Seguridad | Sin autenticación ni límites de tasa | Cualquiera con acceso a la red puede crear campos y sesiones |
| Escalado | Estado en memoria del proceso | Un solo worker; no hay sesiones compartidas |
| Memoria | El historial de sesiones crece sin límite | Conversaciones largas consumen RAM |
| Puntuación | Mensajes evaluados de forma aislada | Sin contexto de varios turnos dentro de la puntuación |
| Calibración | Umbrales de los presets son valores de partida | Conviene medir falsos positivos con datos propios |
| Errores | Fallos del modelo de embeddings no se traducen a un código controlado | Se devuelve `500` genérico |
| Dependencias | `pydantic-settings` usa `class Config` (estilo obsoleto de Pydantic v2) | Aviso de deprecación en los tests; funciona |
| API | `GET /sessions/{id}` no incluye `consecutive_off` ni el historial | Hay que llevar el contador en el cliente, o añadir el campo |
| Licencia | Ninguna | Ver la sección [Licencia](../README.md#licencia) (AGPL-3.0) |

### Discrepancias corregidas en la documentación

El README anterior tenía algunos datos que no coincidían con el código:

- El ejemplo de respuesta de `POST /fields` mostraba `threshold: 0.55` para `moderado`. El valor real es **0.22**.
- Se describían `DEFAULT_*` y `DB_PATH` como si tuvieran efecto. No lo tienen.

## Próximos pasos sugeridos

1. Implementar `repository_sqlite.py` y usar `DB_PATH`.
2. Hacer que `Settings` alimente los valores por defecto de sesión (`patience`, `rolling_window`) y de campo (`alpha`, `top_k`).
3. Incluir `consecutive_off` y, opcionalmente, el historial en `SessionResponse`.
4. Migrar `Settings` a `model_config = SettingsConfigDict(...)`.
6. Calibrar los umbrales de los presets con conversaciones reales del dominio de destino.
