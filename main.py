import hmac
import hashlib
import os
from fastapi import FastAPI, Request, HTTPException
from dotenv import load_dotenv
from agent import review_pr

load_dotenv()

app = FastAPI()

WEBHOOK_SECRET = os.getenv("GITHUB_WEBHOOK_SECRET")

def verify_signature(payload_body, signature_header):

    if not signature_header:
        return False

    expected_signature = "sha256=" + hmac.new(
        WEBHOOK_SECRET.encode(),
        payload_body,
        hashlib.sha256
    ).hexdigest()

    return hmac.compare_digest(expected_signature, signature_header)


@app.get("/")
def home():
    return {"status": "AI PR Review Agent Running"}


@app.post("/webhooks/github")
async def github_webhook(request: Request):

    body = await request.body()

    signature = request.headers.get("X-Hub-Signature-256")

    if not verify_signature(body, signature):
        raise HTTPException(status_code=401, detail="Invalid signature")

    payload = await request.json()

    event = request.headers.get("X-GitHub-Event")

    if event == "pull_request":

        action = payload.get("action")

        if action in ["opened", "synchronize"]:

            pr_number = payload["pull_request"]["number"]
            repo_name = payload["repository"]["full_name"]

            review_pr(repo_name, pr_number)

            return {"message": "Review started"}

    return {"message": "Event ignored"}