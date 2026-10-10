import uvicorn
from llmkeeper.infrastructure.app import create_app
from llmkeeper.infrastructure.config import Settings

if __name__ == "__main__":
    settings = Settings()
    app = create_app(settings)
    uvicorn.run(app, host="127.0.0.1", port=8000)
