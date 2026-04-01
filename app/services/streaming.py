from openai import AsyncOpenAI
from app.config import settings
from app.database import get_collection
from app.services.embedding import create_embedding
from app.services.rag import search_tool, search_documents
import json
import asyncio

client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)


async def stream_rag_response(user_message: str,conversation_history: list | None = None,websocket = None , text_chunk_callback= None):
    if conversation_history is None:
        conversation_history = []
    
    system_instruction = """You are a helpful AI assistant with access to a company knowledge base.
IMPORTANT: You have TWO capabilities:
1. GENERAL KNOWLEDGE - Answer from your training:
   ✓ Math, science, history, geography
   ✓ Programming concepts and explanations
   ✓ Common knowledge questions
2. KNOWLEDGE BASE - Search uploaded documents:
   ✓ Company-specific information
   ✓ Resume/personal details
   ✓ Document content
RULE: Answer directly if it's general knowledge. Only search for specific document/company info.
"""
    openai_messages = [{"role": "system", "content": system_instruction}] + conversation_history

    try:
        if websocket:
            await websocket.send_json({
                "type": "status",
                "message": "Processing your question..."
            })
        
        response = await client.chat.completions.create(
            model=settings.CHAT_MODEL,
            messages=openai_messages,
            tools=[search_tool],
            tool_choice="auto",
            temperature=0.7,
            stream=True
        )
        
        function_calls = []
        full_answer = ""
        used_rag = False
        current_tool_call = None
        tool_call_args = ""
        
        async for chunk in response:
            delta = chunk.choices[0].delta if chunk.choices else None
            if not delta:
                continue

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    if tc.function and tc.function.name:
                        current_tool_call = {
                            "id": tc.id,
                            "name": tc.function.name,
                            "arguments": ""
                        }
                        tool_call_args = ""
                    if tc.function and tc.function.arguments:
                        tool_call_args += tc.function.arguments
                        if current_tool_call:
                            current_tool_call["arguments"] = tool_call_args
            
            if delta.content:
                full_answer += delta.content
                if websocket:
                    await websocket.send_json({"type": "answer", "chunk": delta.content})
                if text_chunk_callback:
                    await text_chunk_callback(delta.content)

            
            # Check finish reason
            if chunk.choices[0].finish_reason == "tool_calls" and current_tool_call:
                function_calls.append(current_tool_call)
                current_tool_call = None
                tool_call_args = ""

        answer = full_answer
        
        if function_calls:
            used_rag = True
            
            if websocket:
                await websocket.send_json({
                    "type": "status",
                    "message": "🔍 Searching documents..."
                })

            for fc in function_calls:
                if fc["name"] == "search_documents":
                    args = json.loads(fc["arguments"]) if fc["arguments"] else {}
                    query = args.get("query", "")
                    search_results = await asyncio.to_thread(search_documents, query)
                
                    conversation_history.append({
                        "role": "assistant",
                        "content": f"I searched the knowledge base for '{query}'."
                    })
                    conversation_history.append({
                        "role": "system",
                        "content": f"Search Results: {search_results}"
                    })

            if websocket:
                await websocket.send_json({
                    "type": "status",
                    "message": "✨ Generating answer..."
                })
            final_messages = [{"role": "system", "content": system_instruction}] + conversation_history

            response_stream = await client.chat.completions.create(
                model=settings.CHAT_MODEL,
                messages=final_messages,
                temperature=0.7,
                stream=True
            )
            
            full_answer = ""
            async for chunk in response_stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    full_answer += delta.content
                    
                    if websocket:
                        await websocket.send_json({
                            "type": "answer",
                            "chunk": delta.content
                        })
                    if text_chunk_callback:
                        await text_chunk_callback(delta.content)
            
            answer = full_answer
        
        conversation_history.append({
            "role": "assistant",
            "content": answer
        })
        
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
            await websocket.send_json({
                "type": "error",
                "message": error_message
            })
        
        return {
            "answer": error_message,
            "used_rag": False,
            "error": str(e)
        }