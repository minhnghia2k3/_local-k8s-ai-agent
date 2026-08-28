from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from kubernetes import client, config
from kubernetes.client.exceptions import ApiException
from botocore.config import Config
import boto3
import asyncio
import os

app = FastAPI(title="Local K8s AI Agent")

app.mount("/static", StaticFiles(directory="static"), name="static")

BEDROCK_REGION = os.getenv("BEDROCK_REGION", "us-east-1")
INFERENCE_PROFILE = os.getenv(
    "BEDROCK_INFERENCE_PROFILE",
    "arn:aws:bedrock:us-east-1:716145636434:inference-profile/global.anthropic.claude-haiku-4-5-20251001-v1:0",
)
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))
MODEL = os.getenv("MODEL", "claude-haiku-4-5")

bedrock = boto3.client(
    "bedrock-runtime",
    region_name=BEDROCK_REGION,
    config=Config(retries={"max_attempts": 5, "mode": "adaptive"}),
)

DEFAULT_NAMESPACE = os.getenv("DEFAULT_NAMESPACE", "ai-devops")

SYSTEM_PROMPT = """You are a DevOps assistant specializing in Kubernetes.
When given an error or question, you:
1. Explain what it means clearly
2. Provide the exact kubectl commands to diagnose or fix it
3. Explain why the fix works
Be concise and practical."""

DIAGNOSE_PROMPT = """You are a Kubernetes diagnostic agent. You are given live cluster state below.
Analyze the actual data and answer the user's question by referencing specific pods, events, or log lines from the state.
If you find problems, suggest the exact kubectl commands to investigate or fix them.
Do not invent pods or errors that are not present in the state."""

try:
    config.load_incluster_config()
    k8s_v1 = client.CoreV1Api()
    K8S_AVAILABLE = True
except config.ConfigException:
    k8s_v1 = None
    K8S_AVAILABLE = False


class Query(BaseModel):
    question: str


class Answer(BaseModel):
    answer: str
    model: str


class DiagnoseRequest(BaseModel):
    question: str
    namespace: str | None = None


class DiagnoseResponse(BaseModel):
    answer: str
    model: str
    context: str


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    return {"status": "ok", "k8s_api_available": K8S_AVAILABLE}


async def call_bedrock(prompt: str, system: str) -> str:
    resp = await asyncio.to_thread(
        bedrock.converse,
        modelId=INFERENCE_PROFILE,
        system=[{"text": system}],
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": MAX_TOKENS},
    )
    return resp["output"]["message"]["content"][0]["text"]


@app.post("/ask", response_model=Answer)
async def ask(query: Query):
    answer = await call_bedrock(query.question, SYSTEM_PROMPT)
    return Answer(answer=answer, model=MODEL)


def gather_namespace_context(namespace: str) -> str:
    if not K8S_AVAILABLE:
        return "Kubernetes API is not available (running outside the cluster)."

    parts = [f"# Cluster state for namespace: {namespace}\n"]

    try:
        pods = k8s_v1.list_namespaced_pod(namespace).items
    except ApiException as e:
        return f"Failed to list pods in '{namespace}': {e.reason}"

    parts.append(f"## Pods ({len(pods)} total)")
    if not pods:
        parts.append("No pods found.")
    for p in pods:
        cs = p.status.container_statuses or []
        restarts = sum(c.restart_count for c in cs)
        waiting = next((c.state.waiting.reason for c in cs if c.state and c.state.waiting), None)
        suffix = f" - waiting: {waiting}" if waiting else ""
        parts.append(f"- {p.metadata.name}: phase={p.status.phase}, restarts={restarts}{suffix}")

    try:
        events = k8s_v1.list_namespaced_event(namespace).items
    except ApiException:
        events = []

    if events:
        parts.append(f"\n## Recent events (last 10)")
        recent = sorted(events, key=lambda e: e.last_timestamp or e.event_time or "", reverse=True)[:10]
        for e in recent:
            parts.append(f"- [{e.type}] {e.reason} ({e.involved_object.kind}/{e.involved_object.name}): {e.message}")

    problem_pods = [p for p in pods if p.status.phase not in ("Running", "Succeeded")]
    for p in problem_pods[:3]:
        try:
            logs = k8s_v1.read_namespaced_pod_log(p.metadata.name, namespace, tail_lines=20)
            parts.append(f"\n## Logs from {p.metadata.name} (last 20 lines)")
            parts.append(logs.strip() or "(no log output)")
        except ApiException:
            pass

    return "\n".join(parts)


@app.post("/diagnose", response_model=DiagnoseResponse)
async def diagnose(req: DiagnoseRequest):
    ns = req.namespace or DEFAULT_NAMESPACE
    context = gather_namespace_context(ns)
    prompt = f"""{context}

---

User question: {req.question}

Analyze the cluster state above and answer the question. Reference specific pods, events, or log lines."""
    answer = await call_bedrock(prompt, DIAGNOSE_PROMPT)
    return DiagnoseResponse(answer=answer, model=MODEL, context=context)

# Now with CI/CD
