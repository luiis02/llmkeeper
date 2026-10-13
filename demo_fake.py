#!/usr/bin/env python3
"""
Demo script with fake embeddings (no model loading).
Shows how LLMKeeper works with concrete, deterministic vectors.
"""

from llmkeeper.infrastructure.config import Settings
from llmkeeper.infrastructure.container import Container
from llmkeeper.domain.models import StrictnessLevel
from tests.fakes import ScriptedEmbeddingProvider


def main():
    print("=" * 80)
    print("🚀 LLMKeeper Demo (con embeddings determinísticos)")
    print("=" * 80)

    # Setup: Define embeddings para los chunks y mensajes
    embeddings_map = {
        # Chunks del system prompt (chunkeados automáticamente)
        "You are a professional data scientist assistant.": [1.0, 0.0, 0.0],
        "Help users with:": [0.9, 0.1, 0.0],
        "Machine learning model selection and training": [0.95, 0.05, 0.0],
        "Data preprocessing and feature engineering": [0.85, 0.15, 0.0],
        "Statistical analysis and hypothesis testing": [0.8, 0.2, 0.0],
        "Python libraries like scikit-learn, pandas, numpy": [0.9, 0.1, 0.0],
        # On-topic messages
        "How do I split my dataset into train and test sets?": [0.92, 0.08, 0.0],
        "What are the best features for predicting house prices?": [0.88, 0.12, 0.0],
        "Can you explain cross-validation?": [0.9, 0.1, 0.0],
        "What is the difference between overfitting and underfitting?": [0.87, 0.13, 0.0],
        # Off-topic messages (cercanos a [0, 0, 1])
        "What is the capital of France?": [0.1, 0.1, 0.8],
        "How do I bake a cake?": [0.15, 0.05, 0.8],
        "Tell me about the history of Rome.": [0.05, 0.15, 0.8],
        "What's your favorite movie?": [0.1, 0.1, 0.8],
        # Session conversation messages
        "How do I normalize features in scikit-learn?": [0.91, 0.09, 0.0],
        "What algorithm should I use for classification?": [0.89, 0.11, 0.0],
        "By the way, what's your favorite movie?": [0.1, 0.1, 0.8],
        "Tell me about Rome's history.": [0.05, 0.15, 0.8],
        "Should I use cross-validation?": [0.9, 0.1, 0.0],
    }

    provider = ScriptedEmbeddingProvider(embeddings_map, default_dim=3)
    settings = Settings()
    container = Container(settings=settings, embedding_provider=provider)

    print("\n" + "=" * 80)
    print("1️⃣  CREAR CAMPO TEMÁTICO desde system prompt")
    print("=" * 80)

    system_prompt = """You are a professional data scientist assistant.
Help users with:
- Machine learning model selection and training
- Data preprocessing and feature engineering
- Statistical analysis and hypothesis testing
- Python libraries like scikit-learn, pandas, numpy"""

    print(f"\nSystem prompt:\n{system_prompt}\n")

    create_field_uc = container.create_field_uc()
    field = create_field_uc.execute(
        system_prompt=system_prompt,
        strictness_preset=StrictnessLevel.MODERADO,
    )

    print(f"✓ Campo creado: {field.field_id}")
    print(f"✓ Número de anchors: {len(field.anchors)}")
    print(f"✓ Threshold (moderado): {field.strictness.threshold}")
    print(f"✓ Dispersión: {field.dispersion:.3f}")
    print(f"✓ Alpha (peso anchors vs centroid): {field.alpha}")

    print("\nAnchors generados:")
    for i, anchor in enumerate(field.anchors, 1):
        print(f"  {i}. {anchor.text[:60]}")

    # Use case 2: Validate messages
    print("\n" + "=" * 80)
    print("2️⃣  VALIDAR MENSAJES (on-topic vs off-topic)")
    print("=" * 80)

    validate_uc = container.validate_message_uc()

    on_topic_msgs = [
        "How do I split my dataset into train and test sets?",
        "What are the best features for predicting house prices?",
        "Can you explain cross-validation?",
        "What is the difference between overfitting and underfitting?",
    ]

    off_topic_msgs = [
        "What is the capital of France?",
        "How do I bake a cake?",
        "Tell me about the history of Rome.",
        "What's your favorite movie?",
    ]

    print("\n📊 ON-TOPIC messages:\n")
    for msg in on_topic_msgs:
        result = validate_uc.execute(field.field_id, msg)
        status = "✅ VÁLIDO" if result.is_valid else "❌ INVÁLIDO"
        print(
            f"{status} | score={result.score:.3f} (threshold={result.threshold})"
        )
        print(f"   Mensaje: {msg}")
        print(f"   Anchor más cercano: {result.nearest_anchor_text[:60]}\n")

    print("📊 OFF-TOPIC messages:\n")
    for msg in off_topic_msgs:
        result = validate_uc.execute(field.field_id, msg)
        status = "✅ VÁLIDO" if result.is_valid else "❌ INVÁLIDO"
        print(
            f"{status} | score={result.score:.3f} (threshold={result.threshold})"
        )
        print(f"   Mensaje: {msg}\n")

    # Use case 3: Adjust strictness
    print("=" * 80)
    print("3️⃣  AJUSTAR STRICTNESS (sin reconstruir vectores)")
    print("=" * 80)

    print(f"\nStrictness actual: {field.strictness.preset.value} → {field.strictness.threshold}")

    adjust_uc = container.adjust_strictness_uc()
    adjust_uc.execute(field.field_id, preset=StrictnessLevel.ESTRICTO)

    field_updated = container.validate_message_uc().fields.get(field.field_id)
    print(f"Strictness nuevo: {field_updated.strictness.preset.value} → {field_updated.strictness.threshold}")

    print("\nRe-validando mensaje borderline con nuevo threshold:")
    msg_borderline = "How do I split my dataset into train and test sets?"
    result_old = validate_uc.execute(field.field_id, msg_borderline)
    print(f"  Con ESTRICTO: {result_old.score:.3f} → {'✅ VÁLIDO' if result_old.is_valid else '❌ INVÁLIDO'}")

    # Use case 4: Session with patience
    print("\n" + "=" * 80)
    print("4️⃣  SESIÓN CONVERSACIONAL con contador de paciencia")
    print("=" * 80)

    # Reset to moderado for this demo
    adjust_uc.execute(field.field_id, preset=StrictnessLevel.MODERADO)

    create_session_uc = container.create_session_uc()
    session = create_session_uc.execute(field.field_id, patience=2, rolling_window=3)
    print(f"\nSesión creada: {session.session_id}")
    print(f"Paciencia: {session.patience} (mensajes consecutivos off-topic antes de alerta)")
    print(f"Rolling window: {session.rolling_window} (para promedios)\n")

    append_uc = container.append_message_uc()

    convo = [
        "How do I normalize features in scikit-learn?",
        "What algorithm should I use for classification?",
        "By the way, what's your favorite movie?",  # Off-topic
        "Tell me about Rome's history.",  # Off-topic again → ALERTA
        "Should I use cross-validation?",  # Back on-topic
    ]

    print("Conversación simulada:\n")
    for i, msg in enumerate(convo, 1):
        session = append_uc.execute(session.session_id, msg)
        last_msg = session.history[-1]

        status_emoji = "✅" if last_msg.is_valid else "❌"
        session_status_emoji = "🟢" if session.status.value == "on_topic" else ("🟡" if session.status.value == "warning" else "🔴")

        print(f"{i}. {status_emoji} '{msg}'")
        print(
            f"   Score: {last_msg.score:.3f} | "
            f"Estado sesión: {session_status_emoji} {session.status.value.upper()} | "
            f"Consecutive off: {session.consecutive_off}/{session.patience}"
        )
        print(f"   Rolling avg: {session.rolling_average():.3f}\n")

    # Summary
    print("=" * 80)
    print("📊 RESUMEN")
    print("=" * 80)
    print(f"""
✓ Se creó un campo temático (topic field) con {len(field.anchors)} anchors
✓ Se validaron {len(on_topic_msgs)} mensajes on-topic → todos ✅
✓ Se validaron {len(off_topic_msgs)} mensajes off-topic → todos ❌
✓ Se ajustó strictness dinámicamente sin reconstruir embeddings
✓ Se simuló una sesión de {len(convo)} mensajes con máquina de estados:
  - on_topic (0 fallos consecutivos)
  - warning (1-{session.patience-1} fallos)
  - off_topic ({session.patience}+ fallos)
✓ Se mantuvieron rolling averages para detectar tendencias

🎯 La validación es semántica (basada en similitud coseno de embeddings),
   no string matching, así que entiende conceptos relacionados.
""")


if __name__ == "__main__":
    main()
