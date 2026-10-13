# Puntuación y construcción del campo temático

Este documento explica **cómo se decide si un mensaje está dentro de tema**: desde el texto del *system prompt* hasta el número final que se compara con el umbral.

## Visión general

```mermaid
flowchart LR
    S[System prompt] --> CH[Chunking]
    CH --> EA[Embeddings de anclas]
    X[extra_topics] -.opcional.-> EA
    EA --> CE[Centroide = media de anclas]
    EA --> DI[Dispersión]
    CE --> DI

    M[Mensaje] --> EM[Embedding del mensaje]
    EM --> TOP[Top-k anclas más similares]
    EA --> TOP
    EM --> CS[Similitud con centroide]
    CE --> CS
    TOP --> SC[score = α·media_top_k + (1−α)·centroide]
    CS --> SC
    SC --> V{score ≥ umbral?}
```

## Construcción del campo temático

Implementado en `domain/services/field_builder.py`.

### 1. Chunking (`split_into_chunks`)

El texto se trocea en **anclas**, que son los fragmentos que representan el tema. El algoritmo elige una de dos estrategias:

1. **Por viñetas**: si el texto tiene al menos **dos líneas** que empiezan por `-`, `*`, `•`, `1.` o `1)`, cada viñeta es un chunk y el marcador se elimina.
2. **Por frases**: en otro caso, el texto se parte tras `.`, `!`, `?`, `¿` o `¡` seguidos de espacio.

Después se filtran los fragmentos:

- Se descartan los que tienen **menos de 3 palabras** (`_MIN_WORDS = 3`).
- Se descartan los **duplicados** sin distinguir mayúsculas (`c.lower()`).
- Si no queda ningún fragmento válido, el campo usa **el texto completo** como única ancla.

Si hay más de **50 chunks** (`_MAX_CHUNKS`), se fusionan pares adyacentes repetidamente hasta quedar por debajo del límite.

#### Ejemplo

```text
Eres un asistente de recetas. Solo respondes sobre cocina.
Ayudas con ingredientes y técnicas.
```

Se parten en tres frases: `"Eres un asistente de recetas."`, `"Solo respondes sobre cocina."` y `"Ayudas con ingredientes y técnicas."`. Las tres tienen 3 o más palabras, así que son tres anclas.

```text
Temas permitidos:
- Recetas de cocina
- Ingredientes y sustituciones
- Técnicas culinarias
```

Aquí hay tres viñetas, así que se usan esas tres líneas. El encabezado `Temas permitidos:` no es viñeta y no se incluye como ancla.

> **Consejo:** para un campo bien definido, usa viñetas con temas concretos. Un *system prompt* largo en prosa también funciona, pero cada frase se convierte en un ancla, y las frases de relleno ("Por favor, sé amable") diluyen el centroide.

### 2. Temas adicionales (`extra_topics`)

Cada elemento de `extra_topics` se añade como ancla más, aunque no aparezca en el *system prompt*. Sirve para cubrir temas que quieres permitir pero que el prompt no menciona. Se embeben en el mismo lote que las anclas del prompt.

### 3. Embeddings

Cada ancla se convierte en un vector de 384 dimensiones con `paraphrase-multilingual-MiniLM-L12-v2`. El adaptador las calcula en lote (`batch_size=32`).

### 4. Centroide (`build_centroid`)

Media aritmética de todos los vectores de anclas:

$$
\vec{c} = \frac{1}{n}\sum_{i=1}^{n} \vec{a}_i
$$

No se normaliza: la similitud coseno ya es invariante a la escala.

### 5. Dispersión (`compute_dispersion`)

Distancia coseno media de cada ancla al centroide:

$$
\text{dispersión} = \frac{1}{n}\sum_{i=1}^{n}\left(1 - \cos(\vec{a}_i, \vec{c})\right)
$$

Un valor bajo indica anclas muy parecidas entre sí (tema homogéneo). Un valor alto indica un campo heterogéneo. **La dispersión se devuelve en la respuesta de creación, pero no interviene en la puntuación.** Es útil para diagnosticar un *system prompt* demasiado disperso.

## Fórmula de puntuación

Implementada en `domain/services/drift_evaluator.py` (`score_against_field`).

Para un mensaje con embedding $\vec{m}$:

1. Se calcula la similitud coseno de $\vec{m}$ con **cada ancla**.
2. Se toman las **top-k** más similares (por defecto $k = 3$) y se promedian: $\overline{s}_k$.
3. Se calcula la similitud con el centroide: $s_c = \cos(\vec{m}, \vec{c})$.
4. Se combinan:

$$
\text{score} = \alpha \cdot \overline{s}_k + (1 - \alpha) \cdot s_c
$$

con $\alpha = 0.7$ por defecto.

5. Se compara con el umbral:

$$
\text{is\_valid} = \text{score} \ge \text{umbral}
$$

### Por qué dos términos

| Término | Captura | Punto débil |
|---------|---------|-------------|
| Top-k de anclas | Coincidencia con un subtema concreto. Un mensaje sobre "sustituciones de huevo" sube aunque el resto del campo no se parezca | Una ancla muy específica puede dar una similitud alta con un mensaje solo superficialmente parecido |
| Centroide | Pertenencia al tema general del campo | Un mensaje válido pero de un subtema lejano al centro puede bajar su puntuación |

Con $\alpha = 0.7$, pesa más la coincidencia con subtemas concretos, pero el centroide evita que una sola palabra compartida dispare la puntuación.

### Ejemplo ilustrativo (valores inventados)

Supón un campo con cinco anclas y un mensaje cuyas tres anclas más similares tienen similitudes $0.80$, $0.70$ y $0.60$, y cuya similitud con el centroide es $0.50$:

$$
\overline{s}_3 = \frac{0.80 + 0.70 + 0.60}{3} = 0.70
$$

$$
\text{score} = 0.7 \cdot 0.70 + 0.3 \cdot 0.50 = 0.49 + 0.15 = 0.64
$$

Con el preset `moderado` (umbral $0.22$), `is_valid = true`.

### Detalles de implementación

- Si el campo tiene menos de $k$ anclas, se promedian las que haya.
- `nearest_anchor_text` es el texto de la ancla con **mayor similitud coseno**, no necesariamente la que más contribuye al score. Es para explicabilidad, no para la decisión.
- La similitud coseno devuelve `0.0` si alguno de los vectores tiene norma cero.

## Niveles de estrictud

Definidos en `domain/models.py` (`StrictnessLevel.threshold`):

| Preset | Umbral | Efecto |
|--------|--------|--------|
| `muy_estricto` | 0.32 | Exige una similitud alta; bloquea desvíos leves |
| `estricto` | 0.27 | |
| `moderado` | 0.22 | **Por defecto** |
| `permisivo` | 0.17 | |
| `muy_permisivo` | 0.12 | Solo bloquea desvíos claros |

Al bajar el umbral, aumentan los mensajes aceptados: menos falsos positivos (mensajes válidos marcados como fuera de tema) y más falsos negativos (desvíos que pasan).

### Umbral personalizado

`strictness_value` acepta cualquier número en $[0, 1]$. Un valor fuera de $[0, 1]$ devuelve `422`. Los presets están entre $0.12$ y $0.32$, así que un valor como $0.9$ marcaría casi todo como fuera de tema. Antes de fijar un umbral propio, mídelo con conversaciones reales de tu dominio.

### Cambiar la estrictud de un campo existente

`PATCH /api/v1/fields/{id}/strictness` cambia el umbral. Ten en cuenta:

- Si no envías ni `preset` ni `custom_value`, el campo vuelve a **`moderado`**.
- Si envías ambos, gana `custom_value`.
- El cambio aplica a los mensajes que se validen a partir de ese momento. Las sesiones guardan el umbral **con el que se evaluó cada mensaje** (`MessageScore.threshold`), así que el historial no se reinterpreta.

## Media móvil de la sesión

`Session.rolling_average()` promedia el `score` de los últimos `rolling_window` mensajes (por defecto 5). Si la sesión no tiene mensajes, devuelve `1.0`.

Es una señal de tendencia, no una decisión. Un descenso sostenido de la media, aunque ningún mensaje haya superado la paciencia, indica que la conversación se va alejando del tema.

## Limitaciones de la puntuación

- Los embeddings son **de propósito general**, no entrenados sobre conversaciones de soporte o de un dominio concreto. Los falsos positivos y negativos dependen del tema.
- El centroide mezcla todas las anclas; un campo con temas muy distintos tiene un centroide "de nadie" y puntúa peor los mensajes de cada subtema. Si tu prompt cubre temas muy dispares, considera crear campos separados.
- La puntuación es **léxico-semántica**: no entiende intención ni contexto de varios turnos. Cada mensaje se evalúa de forma aislada; el contexto solo entra a través de la sesión (contador y media).
- No hay calibración automática de umbrales. Los presets son valores de partida.
