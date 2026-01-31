import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")

MODEL = "gpt-4.1-mini"
PRICING = (0.40, 1.60)  # (input, output) per 1M tokens

AGENTS = {
    "Sceptic": """You are a deeply sceptical analyst. You question everything, 
look for holes in arguments, and always ask "but what if that's wrong?" 
Point out assumptions, demand evidence, and highlight what could go wrong. 
Keep your response concise (2-3 paragraphs max).""",

    "Pedantic Fact-Checker": """You are an obsessively precise fact-checker. 
You care about accuracy down to the smallest detail. Correct any imprecise 
language, note when claims lack sources, and flag anything misleading. 
Keep your response concise (2-3 paragraphs max).""",

    "Common Sense Judge": """You are a practical, down-to-earth judge of ideas. 
You cut through jargon and complexity to ask: does this actually make sense 
in the real world? Value simplicity and practicality over theoretical elegance. 
Keep your response concise (2-3 paragraphs max).""",
}

client = OpenAI(api_key=api_key)


def ask_agents(question: str):
    """Ask all agents the same question and print their responses."""
    in_price, out_price = PRICING
    total_cost = 0
    
    print(f"\n{'=' * 60}")
    print(f"QUESTION: {question}")
    print(f"{'=' * 60}")
    
    for agent_name, system_prompt in AGENTS.items():
        print(f"\n--- {agent_name.upper()} ---\n")
        
        try:
            start = time.perf_counter()
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
            )
            elapsed = time.perf_counter() - start
            
            content = response.choices[0].message.content
            usage = response.usage
            cost = (usage.prompt_tokens * in_price + usage.completion_tokens * out_price) / 1_000_000
            total_cost += cost
            
            print(content)
            print(f"\n[{elapsed:.2f}s | {usage.total_tokens} tokens | ${cost:.6f}]")
            
        except Exception as e:
            print(f"Error: {e}")
    
    print(f"\n{'=' * 60}")
    print(f"TOTAL COST: ${total_cost:.6f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    question = "After 1st April , more than 125 of the more than 150 COVID-19 deaths were from patients aged 70 and above "
    ask_agents(question)