from fastapi import APIRouter, HTTPException, Depends
from llmkeeper.application.use_cases import (
    CreateField,
    ValidateMessage,
    AdjustStrictness,
    CreateSession,
    AppendMessageToSession,
    GetSession,
    FieldNotFoundError,
    SessionNotFoundError,
)
from llmkeeper.adapters.inbound.schemas import (
    CreateFieldRequest,
    FieldResponse,
    ValidationRequestSimple,
    ValidationResponse,
    AdjustStrictnessRequest,
    CreateSessionRequest,
    SessionResponse,
    MessageRequest,
    MessageResponseWithSession,
)


def create_router(
    create_field_uc: CreateField,
    validate_message_uc: ValidateMessage,
    adjust_strictness_uc: AdjustStrictness,
    create_session_uc: CreateSession,
    append_message_uc: AppendMessageToSession,
    get_session_uc: GetSession,
) -> APIRouter:
    router = APIRouter(prefix="/api/v1", tags=["llmkeeper"])

    @router.post("/fields", response_model=FieldResponse)
    async def post_fields(req: CreateFieldRequest) -> FieldResponse:
        field = create_field_uc.execute(
            system_prompt=req.system_prompt,
            extra_topics=req.extra_topics,
            strictness_preset=req.strictness_preset,
            strictness_value=req.strictness_value,
        )
        return FieldResponse(
            field_id=field.field_id,
            system_prompt=field.system_prompt,
            num_anchors=len(field.anchors),
            dispersion=field.dispersion,
            threshold=field.strictness.threshold,
            preset=field.strictness.preset.value if field.strictness.preset else None,
            alpha=field.alpha,
            top_k=field.top_k,
        )

    @router.get("/fields/{field_id}", response_model=FieldResponse)
    async def get_field(field_id: str) -> FieldResponse:
        field = validate_message_uc.fields.get(field_id)
        if field is None:
            raise HTTPException(status_code=404, detail="Field not found")
        return FieldResponse(
            field_id=field.field_id,
            system_prompt=field.system_prompt,
            num_anchors=len(field.anchors),
            dispersion=field.dispersion,
            threshold=field.strictness.threshold,
            preset=field.strictness.preset.value if field.strictness.preset else None,
            alpha=field.alpha,
            top_k=field.top_k,
        )

    @router.patch("/fields/{field_id}/strictness")
    async def patch_strictness(
        field_id: str, req: AdjustStrictnessRequest
    ) -> dict[str, str]:
        try:
            adjust_strictness_uc.execute(
                field_id=field_id, preset=req.preset, custom_value=req.custom_value
            )
        except FieldNotFoundError:
            raise HTTPException(status_code=404, detail="Field not found")
        return {"message": "Strictness updated"}

    @router.post("/fields/{field_id}/validate", response_model=ValidationResponse)
    async def post_validate(
        field_id: str, req: ValidationRequestSimple
    ) -> ValidationResponse:
        try:
            result = validate_message_uc.execute(field_id=field_id, text=req.text)
        except FieldNotFoundError:
            raise HTTPException(status_code=404, detail="Field not found")
        return ValidationResponse(
            is_valid=result.is_valid,
            score=result.score,
            threshold=result.threshold,
            nearest_anchor_text=result.nearest_anchor_text,
        )

    @router.post("/fields/{field_id}/sessions", response_model=SessionResponse)
    async def post_session(
        field_id: str, req: CreateSessionRequest
    ) -> SessionResponse:
        try:
            session = create_session_uc.execute(
                field_id=field_id,
                patience=req.patience,
                rolling_window=req.rolling_window,
            )
        except FieldNotFoundError:
            raise HTTPException(status_code=404, detail="Field not found")
        return SessionResponse(
            session_id=session.session_id,
            field_id=session.field_id,
            patience=session.patience,
            rolling_window=session.rolling_window,
            status=session.status.value,
            rolling_average=session.rolling_average(),
            num_messages=len(session.history),
        )

    @router.post(
        "/sessions/{session_id}/messages", response_model=MessageResponseWithSession
    )
    async def post_message(
        session_id: str, req: MessageRequest
    ) -> MessageResponseWithSession:
        try:
            session = append_message_uc.execute(session_id=session_id, text=req.text)
        except SessionNotFoundError:
            raise HTTPException(status_code=404, detail="Session not found")
        except FieldNotFoundError:
            raise HTTPException(status_code=404, detail="Field not found")
        last_msg = session.history[-1]
        return MessageResponseWithSession(
            score=last_msg.score,
            is_valid=last_msg.is_valid,
            threshold=session.history[-1].score if session.history else 0.0,
            rolling_average=session.rolling_average(),
            status=session.status.value,
            consecutive_off=session.consecutive_off,
        )

    @router.get("/sessions/{session_id}", response_model=SessionResponse)
    async def get_session(session_id: str) -> SessionResponse:
        try:
            session = get_session_uc.execute(session_id=session_id)
        except SessionNotFoundError:
            raise HTTPException(status_code=404, detail="Session not found")
        return SessionResponse(
            session_id=session.session_id,
            field_id=session.field_id,
            patience=session.patience,
            rolling_window=session.rolling_window,
            status=session.status.value,
            rolling_average=session.rolling_average(),
            num_messages=len(session.history),
        )

    return router
