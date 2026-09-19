from fastapi import FastAPI

app = FastAPI(title="NewsAnalyser")


@app.get("/api/health")
def health_check():
    return {"status": "ok"}