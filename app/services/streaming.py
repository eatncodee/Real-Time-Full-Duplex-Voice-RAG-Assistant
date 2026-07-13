from openai import AsyncOpenAI
from app.config import settings
from app.services.rag import search_documents
import asyncio

client = AsyncOpenAI(
    api_key=settings.OPENAI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Keywords that signal the query needs document/resume search.
# Tune this list to match what's actually in your 20 docs.
SEARCH_KEYWORDS = [
    "rudraksh", "resume", "experience", "project", "skill", "education",
    "work", "job", "company", "internship", "certification", "achievement",
    "leetcode", "cgpa", "degree", "college", "university", "background",
    "document", "policy"
]


def needs_search(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in SEARCH_KEYWORDS)


async def stream_rag_response(user_message: str, conversation_history: list | None = None, websocket=None, text_chunk_callback=None):
    if conversation_history is None:
        conversation_history = []

    system_instruction = """You are a voice assistant. The user is listening, not reading — every word costs time.

CAPABILITIES:
1. GENERAL KNOWLEDGE — answer directly from your training
2. KNOWLEDGE BASE — you may receive retrieved document context below a message; use it if relevant

RESPONSE RULES (non-negotiable):
- 1-2 sentences maximum. Never more.
- Answer first, no preamble ("Great question!", "Sure, I can help with that")
- No lists, no bullet points, no markdown, no headers
- Plain spoken language only — this gets read aloud by TTS
- If the full answer needs more detail, give the single most important fact and stop. Do not summarize everything you know.

Example:
User: "Is Rudraksh a good problem solver?"
Bad: "Based on the resume, Rudraksh has solved over 350 LeetCode problems with a rating of 1700, has worked on several projects including a voice RAG assistant, and has experience with FastAPI, Docker, and MongoDB, which indicates strong..."
Good: "Yes — he's solved 350+ LeetCode problems at a 1700 rating, which is a strong signal."
"""

    used_rag = False
    search_message_index = None  # track so we can strip it after use — see note below

    # --- Eager retrieval: decide BEFORE calling the LLM, no tool-call round trip ---
    if needs_search(user_message):
        used_rag = True
        if websocket:
            await websocket.send_json({"type": "status", "message": "🔍 Searching documents..."})

        search_results = await asyncio.to_thread(search_documents, user_message)

        conversation_history.append({
            "role": "system",
            "content": f"Search Results: {search_results}\n\nReminder: answer in 1-2 sentences maximum, plain language, no lists, no bold, no preamble. State only the single most relevant fact."
        })
        search_message_index = len(conversation_history) - 1

    openai_messages = [{"role": "system", "content": system_instruction}] + conversation_history

    try:
        if websocket:
            await websocket.send_json({"type": "status", "message": "✨ Generating answer..."})

        response = await client.chat.completions.create(
            model=settings.CHAT_MODEL,
            messages=openai_messages,
            temperature=0.7,
            stream=True
        )

        full_answer = ""
        async for chunk in response:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                full_answer += delta.content
                if websocket:
                    await websocket.send_json({"type": "answer", "chunk": delta.content})
                if text_chunk_callback:
                    await text_chunk_callback(delta.content)

        answer = full_answer

        # Remove the raw search-results dump now that it's served its purpose
        # for THIS turn's generation. Without this, every RAG turn permanently
        # bloats conversation_history with a full document forever (since
        # search_documents() now returns entire documents, not just top-3
        # chunks) — across a multi-turn session this compounds fast and lets
        # stale/irrelevant documents from earlier questions leak into later
        # answers. The concise final answer we append below is enough
        # persisted memory for natural follow-ups.
        if search_message_index is not None:
            conversation_history.pop(search_message_index)

        conversation_history.append({"role": "assistant", "content": answer})

        if websocket:
            await websocket.send_json({
                "type": "done",
                "used_rag": used_rag,
                "conversation_history": conversation_history
            })

        return {
            "answer": answer,
            "used_rag": used_rag,
            "conversation_history": conversation_history
        }

    except Exception as e:
        error_message = f"Error: {str(e)}"
        if websocket:
            await websocket.send_json({"type": "error", "message": error_message})
        return {
            "answer": error_message,
            "used_rag": False,
            "error": str(e)
        }