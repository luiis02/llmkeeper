#!/usr/bin/env python3
"""
Demo script: Use LLMKeeper programmatically (without server).
Shows how to create a topic field and validate messages.
"""

from llmkeeper.infrastructure.config import Settings
from llmkeeper.infrastructure.container import Container
from llmkeeper.adapters.outbound.embedding_sentence_transformers import (
    SentenceTransformerEmbeddingAdapter,
)
from llmkeeper.domain.models import StrictnessLevel


def main():
    print("🚀 LLMKeeper Demo\n")

    # Initialize
    settings = Settings()
    embedding_provider = SentenceTransformerEmbeddingAdapter(
        model_name=settings.embedding_model_name
    )
    container = Container(settings=settings, embedding_provider=embedding_provider)

    # Use case 1: Create a topic field
    print("1️⃣ Creating topic field from system prompt...")
    system_prompt = """
    You are a professional data scientist assistant.
    Help users with:
    - Machine learning model selection and training
    - Data preprocessing and feature engineering
    - Statistical analysis and hypothesis testing
    - Python libraries like scikit-learn, pandas, numpy
    """

    create_field_uc = container.create_field_uc()
    field = create_field_uc.execute(
        system_prompt=system_prompt,
        strictness_preset=StrictnessLevel.MODERADO,
    )
    print(f"   ✓ Field created: {field.field_id}")
    print(f"   ✓ Anchors: {len(field.anchors)}")
    print(f"   ✓ Threshold: {field.strictness.threshold}\n")

    # Use case 2: Validate messages
    print("2️⃣ Validating messages...\n")

    validate_uc = container.validate_message_uc()

    on_topic_msgs = [
        "How do I split my dataset into train and test sets?",
        "What are the best features for predicting house prices?",
        "Can you explain cross-validation?",
    ]

    off_topic_msgs = [
        "What is the capital of France?",
        "How do I bake a cake?",
        "Tell me about the history of Rome.",
    ]

    print("   ON-TOPIC messages:")
    for msg in on_topic_msgs:
        result = validate_uc.execute(field.field_id, msg)
        status = "✓" if result.is_valid else "✗"
        print(f"   {status} {msg[:50]:<50} | score={result.score:.3f}")

    print("\n   OFF-TOPIC messages:")
    for msg in off_topic_msgs:
        result = validate_uc.execute(field.field_id, msg)
        status = "✓" if result.is_valid else "✗"
        print(f"   {status} {msg[:50]:<50} | score={result.score:.3f}")

    # Use case 3: Session with patience
    print("\n3️⃣ Simulating conversation session...\n")

    create_session_uc = container.create_session_uc()
    session = create_session_uc.execute(field.field_id, patience=2, rolling_window=3)
    print(f"   Session created: {session.session_id}")
    print(f"   Patience: {session.patience} (off-topic messages before alert)\n")

    append_uc = container.append_message_uc()

    convo = [
        ("What ML model should I use?", True),
        ("How do I normalize features?", True),
        ("By the way, what's your favorite pizza?", False),
        ("Can you recommend a good ML course?", True),
    ]

    for msg, expected_valid in convo:
        session = append_uc.execute(session.session_id, msg)
        emoji = "✓" if session.history[-1].is_valid == expected_valid else "⚠"
        print(
            f"   {emoji} '{msg}'"
            f" | valid={session.history[-1].is_valid}, "
            f"status={session.status.value}, "
            f"rolling_avg={session.rolling_average():.2f}"
        )

    print("\n✅ Demo complete!")


if __name__ == "__main__":
    main()
