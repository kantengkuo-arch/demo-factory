"""
{{PROJECT_NAME}} - Backend API
{{DESCRIPTION}}
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="{{PROJECT_NAME}}", description="{{DESCRIPTION}}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "{{PROJECT_NAME}} is running", "status": "ok"}


@app.get("/health")
def health():
    return {"status": "healthy"}


# === 在下方添加你的 API 路由 ===


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
