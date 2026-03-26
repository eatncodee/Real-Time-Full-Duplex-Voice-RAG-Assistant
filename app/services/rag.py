from openai import OpenAI
from app.config import settings
from app.database import get_collection
from app.services.embedding import create_embedding,create_embeddings_batch
import json


client = OpenAI(api_key=settings.OPENAI_API_KEY)


search_tool = {
    "type": "function",
    "function": {
        "name": "search_documents",
        "description": """Search the internal knowledge base for specific factual information.

USE this function ONLY when the user asks about:

📊 Company/Organization Information:
- Financial data (revenue, growth, metrics, budgets)
- Employee count, team structure, hiring information
- Products, services, features, or offerings
- Policies, procedures, guidelines, or documentation
- Office locations, company history, founding details
- Leadership (CEO, executives, managers)

👤 Personal Information (from uploaded resumes/documents):
- Education background, degrees, universities
- Work experience, job history, previous roles
- Skills, certifications, qualifications
- Projects, achievements, publications
- Contact information, languages spoken
- Any biographical or professional details about individuals mentioned in uploaded documents

📄 Document Content:
- Specific facts, dates, or details from uploaded files
- Quotes or excerpts from documents
- Technical specifications or requirements
- Meeting notes, action items, decisions

DO NOT use this function for:

❌ General Knowledge:
- Math calculations (2+2, 50*30, etc.)
- Scientific facts (photosynthesis, gravity, etc.)
- Historical events (World War II, etc.)
- Geography (capitals, countries, etc.)
- Common definitions or explanations

❌ Creative Tasks:
- Writing stories, poems, or code
- Brainstorming ideas
- Giving opinions or advice (unless based on uploaded documents)

❌ Conversational:
- Greetings (Hi, Hello, How are you)
- Casual chat
- Clarification questions

🎯 Decision Rule:
If the answer could be in an uploaded document or company knowledge base → SEARCH
If the answer is general knowledge or conversational → DO NOT SEARCH

Examples that NEED search:
✅ "What was our Q4 revenue?"
✅ "What is Rudraksh's educational background?"
✅ "What skills does the uploaded resume mention?"
✅ "How many employees do we have?"
✅ "What projects has the person worked on?"
✅ "What is mentioned about Python in the documents?"

Examples that DON'T need search:
❌ "What is 2+2?"
❌ "Explain what Python is"
❌ "What's the capital of France?"
❌ "Write a function to sort an array"
❌ "Hello, how are you?"
❌ "Can you help me?" """,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find relevant documents"
                }
            },
            "required": ["query"]
        }
    }
}


def search_documents(question: str)-> str:
    collection = get_collection()
    
    q_emb = create_embedding(question)
    if q_emb:
        results = collection.query(
            query_embeddings=[q_emb],
            n_results=3
        )
        
        if results['documents'] and len(results['documents']) > 0:
            docs=results['documents'][0]
            return "\n\n".join(docs)
    return "No relevant docs found"


def _build_openai_messages(conversation_history: list, system_instruction: str) -> list:
    """Convert conversation history to OpenAI message format."""
    messages = [{"role": "system", "content": system_instruction}]
    for entry in conversation_history:
        role = entry["role"]
        if role == "model":
            role = "assistant"
        
        parts = entry.get("parts", [])
        for part in parts:
            if isinstance(part, dict):
                if "text" in part:
                    messages.append({"role": role, "content": part["text"]})
                elif "function_call" in part:
                    fc = part["function_call"]
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": f"call_{fc['name']}",
                            "type": "function",
                            "function": {
                                "name": fc["name"],
                                "arguments": json.dumps(fc.get("args", {}))
                            }
                        }]
                    })
                elif "function_response" in part:
                    fr = part["function_response"]
                    messages.append({
                        "role": "tool",
                        "tool_call_id": f"call_{fr['name']}",
                        "content": json.dumps(fr["response"])
                    })
            elif isinstance(part, str):
                messages.append({"role": role, "content": part})
    return messages


def chat_with_function_calling(user_message: str, conversation_history:list | None = None,temprature: float=0.7) -> dict:
    if conversation_history is None:
        conversation_history = []
    conversation_history.append({"role": "user","parts": [{"text":user_message}]})   
    system_instruction = """You are a helpful AI assistant with access to a company knowledge base.

IMPORTANT: You have TWO capabilities:
1. GENERAL KNOWLEDGE - Answer from your training
2. KNOWLEDGE BASE - Search uploaded documents

RESPONSE STYLE:
-give answer in short no big pragraphs
- Be concise and direct
- Answer the question FIRST, then provide brief supporting evidence
- Use bullet points only when listing 3+ items
- For yes/no questions, lead with yes/no
- Keep responses focused on what was asked

Example:
User: "Is Rudraksh at a good learning pace?"
Bad: [Lists all skills, projects, achievements] "Therefore, yes he is at a good pace"
Good: "Yes, Rudraksh is at an excellent learning pace for a 6th semester student. Key indicators: 8.2 CGPA, 350+ LeetCode problems (1700 rating), and hands-on projects with modern tech (FastAPI, Docker, MongoDB). He's well ahead of typical peers."

RULE: Answer directly if it's general knowledge. Only search for specific document/company info.
"""
    try:
        messages = _build_openai_messages(conversation_history, system_instruction)
        
        response = client.chat.completions.create(
            model=settings.CHAT_MODEL,
            messages=messages,
            tools=[search_tool],
            tool_choice="auto",
            temperature=temprature
        )

        choice = response.choices[0]
        
        if choice.message.tool_calls:
            function_calls = choice.message.tool_calls
            
            for fc in function_calls:
                if fc.function.name == "search_documents":
                    query = json.loads(fc.function.arguments).get("query", "")
                    search_results = search_documents(query)
                    
                    conversation_history.append({"role": "model","parts": [{"function_call": {"name": fc.function.name, "args": json.loads(fc.function.arguments)}}]})
                    conversation_history.append({"role": "user","parts": [{"function_response": {"name": "search_documents","response": {"result": search_results}}}]}) 

            final_messages = _build_openai_messages(conversation_history, system_instruction)
            final_response = client.chat.completions.create(
                model=settings.CHAT_MODEL,
                messages=final_messages,
                temperature=temprature
            )
            
            answer = final_response.choices[0].message.content
            used_rag = True
        
        else:
            answer = choice.message.content
            used_rag = False
        
        conversation_history.append({"role": "model","parts": [{"text":answer}]})
            
        return {
            "answer": answer,
            "used_rag": used_rag,
            "conversation_history": conversation_history,
            "temprature":temprature
        }
    except Exception as e:
        error_message = f"I encountered an error: {str(e)}"
        
        conversation_history.append({
            "role": "model",
            "parts": [{"text": error_message}]
        })
        
        return {
            "answer": error_message,
            "used_rag": False,
            "conversation_history": conversation_history,
            "error": str(e)
        }

def generate_answer(question :str ,context_docs: str):
    prompt = f"""You are a helpful assistant answering questions.
            Context (retrieved information):
            {context_docs}

            Question: {question}

            Instructions:
            - Keep the answer concise and direct

            Answer:"""
    response = client.chat.completions.create(
        model=settings.CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def ask_question(question: str, n_results: int = 3) -> dict:
    retrieved_docs = search_documents(question)
    if not retrieved_docs:
        return {
            'question': question,
            'answer': "I don't have any documents to answer your question. Please upload some documents first.",
            'sources': []
        }
    
    answer = generate_answer(question, retrieved_docs)
    
    return {
        'question': question,
        'answer': answer,
    }