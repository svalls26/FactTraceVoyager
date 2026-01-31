import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")

MODEL = "gpt-4.1-mini"
PRICING = (0.40, 1.60)
DEBATE_DURATION = 15  # seconds

AGENTS = {
    "Sceptic": """You find problems with claims. Look for exaggeration, missing context, or misleading framing.
Quote specific parts. Say FAITHFUL or MUTATION at the end. Keep it to 3-4 sentences.""",

    "Defender": """You defend reasonable interpretations. Is the core meaning preserved despite simplification?
Quote specific parts. Say FAITHFUL or MUTATION at the end. Keep it to 3-4 sentences.""",
}

JURY_PROMPT = """You are a jury deciding if a claim accurately represents a fact.
Give: VERDICT (FAITHFUL/MUTATION), CONFIDENCE (0-100%), and a 1-2 sentence SUMMARY."""

client = OpenAI(api_key=api_key)


def get_agent_response(agent_name: str, system_prompt: str, messages: list) -> dict:
    """Get a response from an agent."""
    in_price, out_price = PRICING
    
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system_prompt}] + messages,
    )
    elapsed = time.perf_counter() - start
    
    content = response.choices[0].message.content
    usage = response.usage
    cost = (usage.prompt_tokens * in_price + usage.completion_tokens * out_price) / 1_000_000
    
    return {
        "agent": agent_name,
        "content": content,
        "time": elapsed,
        "tokens": usage.total_tokens,
        "cost": cost,
    }


def run_debate(internal_fact: str, external_claim: str):
    """Run a timed debate between agents about claim faithfulness."""
    total_cost = 0
    debate_transcript = []
    
    initial_prompt = f"""FACT: "{internal_fact}"
CLAIM: "{external_claim}"

Is the claim faithful to the fact?"""

    print(f"\n{'=' * 70}")
    print("CLAIM VERIFICATION DEBATE")
    print(f"{'=' * 70}")
    print(f"\nFACT: {internal_fact}")
    print(f"\nCLAIM: {external_claim}")
    print(f"\n{'=' * 70}")
    print(f"DEBATE ({DEBATE_DURATION} seconds)")
    print(f"{'=' * 70}")
    
    debate_start = time.time()
    round_num = 1
    conversation_history = [{"role": "user", "content": initial_prompt}]
    
    agent_names = ["Sceptic", "Defender"]
    current_agent_idx = 0
    
    while (time.time() - debate_start) < DEBATE_DURATION:
        agent_name = agent_names[current_agent_idx]
        system_prompt = AGENTS[agent_name]
        
        print(f"\n--- ROUND {round_num}: {agent_name.upper()} ---\n")
        
        try:
            result = get_agent_response(agent_name, system_prompt, conversation_history)
            total_cost += result["cost"]
            
            print(result["content"])
            print(f"\n[{result['time']:.2f}s | {result['tokens']} tokens | ${result['cost']:.6f}]")
            
            debate_transcript.append(f"[{agent_name}]: {result['content']}")
            conversation_history.append({"role": "assistant", "content": result["content"]})
            
            if (time.time() - debate_start) < DEBATE_DURATION:
                next_agent = agent_names[(current_agent_idx + 1) % 2]
                conversation_history.append({
                    "role": "user", 
                    "content": f"Respond to {agent_name}'s points."
                })
            
        except Exception as e:
            print(f"Error: {e}")
            break
        
        current_agent_idx = (current_agent_idx + 1) % 2
        round_num += 1
        
        elapsed = time.time() - debate_start
        if elapsed >= DEBATE_DURATION:
            print(f"\n[Time limit reached: {elapsed:.1f}s]")
            break
    
    print(f"\n{'=' * 70}")
    print("JURY DELIBERATION")
    print(f"{'=' * 70}\n")
    
    jury_prompt = f"""FACT: "{internal_fact}"
CLAIM: "{external_claim}"

DEBATE:
{chr(10).join(debate_transcript)}

Verdict?"""

    try:
        result = get_agent_response(
            "Jury", 
            JURY_PROMPT, 
            [{"role": "user", "content": jury_prompt}]
        )
        total_cost += result["cost"]
        
        print(result["content"])
        print(f"\n[{result['time']:.2f}s | {result['tokens']} tokens | ${result['cost']:.6f}]")
        
    except Exception as e:
        print(f"Jury Error: {e}")
    
    print(f"\n{'=' * 70}")
    print(f"TOTAL COST: ${total_cost:.6f}")
    print(f"{'=' * 70}")

def run_incremental_debate(internal_fact: str, full_claim: str):
    # Split the claim by sentences (simple split for demonstration)
    sentences = [s.strip() for s in full_claim.split('.') if s.strip()]
    cumulative_claim = ""
    total_cost = 0
    
    print(f"FACT BASE: {internal_fact}\n")

    for i, sentence in enumerate(sentences):
        cumulative_claim += sentence + ". "
        print(f"\n{'#' * 70}")
        print(f"PROCESSING SEGMENT {i+1}: {sentence}")
        print(f"{'#' * 70}")

        # The context for this round includes the fact and the claim-so-far
        round_context = f"FACT: {internal_fact}\nCLAIM SO FAR: {cumulative_claim}\nCURRENT SEGMENT: {sentence}"

        # 1. Sceptic looks for the immediate lie/exaggeration in this sentence
        sceptic_task = [{"role": "user", "content": f"{round_context}\nSceptic, what is wrong with this specific segment?"}]
        s_res = get_agent_response("Sceptic", AGENTS["Sceptic"], sceptic_task)
        
        # 2. Defender tries to justify the segment based on the fact
        defender_task = [{"role": "user", "content": f"{round_context}\nSceptic said: {s_res['content']}\nDefender, justify this segment."}]
        d_res = get_agent_response("Defender", AGENTS["Defender"], defender_task)

        print(f"\n[SCEPTIC]: {s_res['content']}")
        print(f"\n[DEFENDER]: {d_res['content']}")

        # 3. Intermediate Jury Verdict for this segment
        jury_task = [{"role": "user", "content": f"{round_context}\nDebate:\nS: {s_res['content']}\nD: {d_res['content']}\nVerdict for this segment?"}]
        j_res = get_agent_response("Jury", JURY_PROMPT, jury_task)
        
        print(f"\n--- SEGMENT {i+1} VERDICT ---\n{j_res['content']}")
        
        total_cost += s_res['cost'] + d_res['cost'] + j_res['cost']
        time.sleep(1) # Visual pacing

    print(f"\n{'=' * 70}\nFINAL SESSION COST: ${total_cost:.6f}\n{'=' * 70}")



if __name__ == "__main__":
    # internal_fact = """As of April 1 , 103 of 122 ( 84 % ) COVID-19 deaths were in patients aged 70 or older , and no one younger than 50 was known to have died from the disease in Massachusetts ."""

    # external_claim = """After 1st April , more than 125 of the more than 150 COVID-19 deaths were from patients aged 70 and above ."""

    # run_debate(internal_fact, external_claim)

    fact = "As of April 1, 103 of 122 COVID-19 deaths were in patients aged 70 or older in MA."
    claim = "After 1st April, more than 125 deaths occurred. Most were 70 and above. This proves the youth were safe."
    run_incremental_debate(fact, claim)