# Detección de prompt injections por dirección vectorial

Pipeline en `scripts/injection_direction.py` que aprende una "dirección de inyección" a partir de dos datasets públicos de prompt injections y evalúa un umbral de detección optimizado con PuLP.

## Qué hace

1. **Datasets.** Carga y une todos los splits de:
   - `deepset/prompt-injections` (662 prompts)
   - `S-Labs/prompt-injection-dataset` (15.291 filas)

   Tras deduplicar por texto se quedan 15.794 prompts únicos: 7.038 inyecciones (etiqueta 1) y 8.756 no inyecciones (etiqueta 0). Se eliminan 158 duplicados exactos y 1 texto con etiquetas contradictorias. Ambos datasets no comparten ningún texto.
2. **Split 80/20 aleatorio.** Reparto con `random.Random(42)`: 12.635 prompts para entrenar y 3.159 para evaluar. No es estratificado.
3. **Dirección vectorial.** Cada prompt se codifica con `paraphrase-multilingual-MiniLM-L12-v2` (el mismo modelo que usa el proyecto) y el embedding se normaliza a norma 1. La dirección de un prompt es ese vector unitario.
4. **Base de datos vectorial (80%).** Se filtran solo las inyecciones del conjunto de entrenamiento (5.629) y sus direcciones se guardan en una colección de ChromaDB con distancia coseno. Las inserciones se hacen en lotes de 4.096 porque Chroma limita el tamaño de cada `add()`. La BD persiste en `.cache/injection_direction/`.
5. **Búsqueda del vecino más cercano (20%).** Para cada prompt de evaluación se consulta la BD y se obtiene la distancia coseno a la inyección más cercana, `d_i = 1 − cos(θ)`.
6. **Optimización con PuLP (MILP).** Se busca un umbral `t` y se clasifica como inyección todo prompt con `d_i ≤ t`:
   - Variables: `t` continua en el rango de distancias y `z_i` binaria por prompt, con `z_i = 1 ⇔ d_i ≤ t` modelado con restricciones de big-M.
   - Objetivo: maximizar `TP − FP`, es decir, inyecciones encontradas menos no inyecciones marcadas como inyecciones.
   - Restricción: `FP ≤ ⌊0.05 · N_test⌋ = 157`.

## Resultados

Ejecución con semilla 42, ambos datasets y `paraphrase-multilingual-MiniLM-L12-v2`:

| Métrica | Valor |
|---|---|
| Umbral óptimo (distancia coseno) | 0.2836 |
| Inyecciones encontradas (TP) | 948 de 1.409 |
| Inyecciones no detectadas (FN) | 461 |
| No inyecciones marcadas (FP) | 157 de 1.750 (5.0% de los prompts de test) |
| No inyecciones correctas (TN) | 1.593 |
| Recall | 0.673 |
| Precisión | 0.858 |

El límite de FP se cumple exactamente (157 ≤ 157).

Resultados de la ejecución anterior, solo con `deepset/prompt-injections` (132 prompts de test): umbral 0.1691, recall 0.462, precisión 0.828, 5 FP.

## Requisitos

Además de las dependencias del proyecto (`sentence-transformers`, `torch`, `numpy`), el script necesita:

```bash
pip install datasets pulp chromadb
```

Estas tres no están declaradas en `pyproject.toml`; están instaladas en el `.venv` del proyecto.

## Ejecución

```bash
.venv/bin/python scripts/injection_direction.py
```

Para usar otro modelo de embeddings sin editar el script:

```bash
INJECTION_EMBED_MODEL=<id-del-modelo> .venv/bin/python scripts/injection_direction.py
```

La primera ejecución descarga los datasets y el modelo desde Hugging Face. Cada ejecución recrea la colección de ChromaDB. Tarda unos 3 minutos en CPU.

## Configuración

Las constantes están al inicio del script:

| Constante | Valor | Significado |
|---|---|---|
| `MODEL_NAME` | `paraphrase-multilingual-MiniLM-L12-v2` (o la variable `INJECTION_EMBED_MODEL`) | Modelo de embeddings |
| `DATASETS` | `deepset/prompt-injections`, `S-Labs/prompt-injection-dataset` | Datasets de Hugging Face |
| `SEED` | `42` | Semilla del split |
| `TRAIN_RATIO` | `0.8` | Proporción de entrenamiento |
| `MAX_FP_RATIO` | `0.05` | Máximo de falsos positivos sobre los prompts de evaluación |
| `DB_DIR` | `.cache/injection_direction/` | Ubicación de la BD vectorial |

## Modelos probados

- `paraphrase-multilingual-MiniLM-L12-v2`: funciona. Es el modelo usado en los resultados.
- `google/embeddinggemma-2`: existe en Hugging Face (Apache-2.0), pero no se puede cargar con el stack actual. Su procesador es multimodal (audio, imagen, vídeo) y el transformers 5.19 del `.venv` no incluye la clase `EmbeddingGemma2Processor`. Requiere una versión de transformers más reciente, sin verificar su impacto en el proyecto. No se han medido sus resultados.

## Decisiones de diseño

- **El 5% se aplica al 20% de evaluación**, no al total de 15.794 prompts.
- **El objetivo combina dos criterios con peso igual.** Maximizar TP y minimizar FP se resume en `TP − FP`. Cambiar los pesos o maximizar TP con FP acotado es una modificación sencilla de la función `solve_threshold`.
- **Solo se usan inyecciones en la BD.** Los prompts seguros no se almacenan; la clasificación depende únicamente de la distancia a la inyección más cercana.
- **Deduplicación antes del split.** Evita que un mismo texto aparezca a la vez en train y en test.

## Limitaciones

- **Sesgo optimista en las métricas.** El umbral se optimiza sobre el mismo 20% con el que se reportan recall y precisión, así que esas cifras están sobreestimadas. Una estimación honesta necesitaría un tercer conjunto o validación cruzada.
- **Recall moderado.** Un único umbral sobre la distancia a la inyección más cercana detecta un 67% de las inyecciones. Las que quedan lejos de todas las inyecciones de entrenamiento no se detectan.
- **Una sola partición.** Los resultados corresponden a la semilla 42. No hay intervalos de confianza ni repeticiones con otras semillas.
- **Distribución de datasets.** Los dos datasets pueden diferir en estilo (uno es más corto y de origen distinto). El recall medido sobre la mezcla no dice nada del rendimiento por fuente.

## Posibles mejoras

- Usar también la distancia a las no inyecciones (por ejemplo, `d_inj − d_safe`) como criterio de decisión.
- Validación cruzada o varias semillas, con la media y la desviación del recall y de los FP.
- Informar métricas por dataset de origen.
- Probar `google/embeddinggemma-2` cuando el entorno pueda cargarlo.
- Un clasificador entrenado sobre los mismos embeddings como línea base.
