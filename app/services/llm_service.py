from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from typing import List, Dict
import json

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

async def get_recommendations(predictions: List[Dict]) -> Dict[str, str]:
    """
    Generate AI-based traffic recommendations using LangChain + OpenAI.

    Args:
        predictions: List of dicts like [{"zone": "Zone 1", "predicted_count": 85}]

    Returns:
        Dict[str, str]: {"Zone 1": "Recommendation text", ...}
    """
    # Convert predictions to string JSON for the prompt
    preds_json = json.dumps(predictions, indent=2)

    # Build structured prompt
    prompt_template = ChatPromptTemplate.from_template("""
You are an AI traffic analyst. Given the predicted traffic counts for each zone, provide practical recommendations.
Be concise, specific, and actionable.

Use the following rules:
- If traffic is high (>80), suggest rerouting or extra public transport.
- If traffic is low (<40), suggest promoting travel during this time.
- Otherwise, suggest monitoring and adjusting signals.

Predictions:
{preds}

Respond **strictly in JSON format** like:
{{
  "Zone 1": "Recommendation...",
  "Zone 2": "Recommendation...",
  "Zone 3": "Recommendation..."
}}
Do not add any text outside of JSON.
""")

    prompt = prompt_template.format_messages(preds=preds_json)

    # Run inference
    response = await llm.ainvoke(prompt)

    # Parse JSON safely
    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        # fallback: return plain text mapped to zones
        return {p["zone"]: response.content for p in predictions}
