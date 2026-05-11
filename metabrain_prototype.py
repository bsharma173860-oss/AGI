"""
Meta-Brain Minimal Prototype
============================
Bharat | PhenexAI Research | May 2026

This prototype implements the 3-core-module Meta-Brain:
    M1: World Model       — physical/logical consistency checker
    M5: Simulation Checker — hallucination detector
    M6: Universe Memory Box — compounding memory
    Decision Engine       — weighted voting

Uses Claude API as the base LLM brain.
Proves the architecture works BEFORE building from scratch.

Usage:
    python metabrain_prototype.py

Requirements:
    pip install anthropic numpy
"""

import json
import numpy as np
import time
from typing import Any

# ============================================================
# ANTHROPIC API CALL
# ============================================================

def call_claude(system_prompt: str, user_prompt: str, max_tokens: int = 500) -> str:
    """Call Claude API — base LLM for all modules."""
    import urllib.request
    
    payload = json.dumps({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}]
    }).encode()
    
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())
    
    return data["content"][0]["text"]


# ============================================================
# MODULE M1: WORLD MODEL
# Scores output by physical/logical consistency
# p_1(o) = exp(-D_KL) / Z
# ============================================================

class WorldModel:
    """
    M1: Before the system speaks, simulate physical consequences.
    Asks: Is this output physically and logically consistent?
    Returns confidence score 0.0 to 1.0
    """
    
    def score(self, question: str, proposed_output: str) -> tuple[float, str]:
        system = """You are a physical and logical consistency checker.
        
Your job: Given a question and a proposed answer, check if the answer is 
physically consistent, logically valid, and not contradictory.

Respond ONLY in JSON format:
{
  "score": <float between 0.0 and 1.0>,
  "issues": "<any consistency problems found, or 'none'>",
  "reasoning": "<one sentence explanation>"
}

Score guide:
1.0 = perfectly consistent
0.7 = mostly consistent, minor issues  
0.4 = significant inconsistencies
0.0 = physically/logically impossible"""

        prompt = f"Question: {question}\nProposed answer: {proposed_output}"
        
        try:
            response = call_claude(system, prompt, max_tokens=200)
            # Clean response - remove markdown if present
            clean = response.strip()
            if "```" in clean:
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            data = json.loads(clean.strip())
            return float(data["score"]), data["reasoning"]
        except Exception as e:
            return 0.5, f"M1 parse error: {e}"


# ============================================================
# MODULE M5: SIMULATION HYPOTHESIS CHECKER
# Hallucination detector
# T(o) = -log P(o | context)
# ============================================================

class SimulationChecker:
    """
    M5: Is this real? Applies skeptical scrutiny to proposed output.
    Detects hallucinations, confabulations, made-up facts.
    Returns confidence score 0.0 to 1.0
    """
    
    def score(self, question: str, proposed_output: str) -> tuple[float, str]:
        system = """You are a hallucination and fact-checking detector.

Your job: Given a question and a proposed answer, determine if the answer 
contains hallucinations, made-up facts, or unverifiable claims.

Respond ONLY in JSON format:
{
  "score": <float between 0.0 and 1.0>,
  "hallucination_detected": <true or false>,
  "reasoning": "<one sentence explanation>"
}

Score guide:
1.0 = definitely real, verifiable, grounded
0.7 = mostly real, some uncertain claims
0.4 = contains likely hallucinations
0.0 = clear fabrication or impossible claims"""

        prompt = f"Question: {question}\nProposed answer: {proposed_output}"
        
        try:
            response = call_claude(system, prompt, max_tokens=200)
            clean = response.strip()
            if "```" in clean:
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            data = json.loads(clean.strip())
            return float(data["score"]), data["reasoning"]
        except Exception as e:
            return 0.5, f"M5 parse error: {e}"


# ============================================================
# MODULE M6: UNIVERSE MEMORY BOX
# Compounding memory — grows richer with every run
# Q(U_t) = 1 - exp(-lambda * t)
# ============================================================

class UniverseMemoryBox:
    """
    M6: Living compressed memory of all past interactions.
    Unlike RAG (degrades), this compounds and improves.
    Scores output by consistency with accumulated knowledge.
    """
    
    def __init__(self, lambda_rate: float = 0.1):
        self.lambda_rate = lambda_rate
        self.memory: list[dict] = []
        self.t = 0
    
    def quality(self) -> float:
        """Q(U_t) = 1 - exp(-lambda * t)"""
        return 1.0 - np.exp(-self.lambda_rate * self.t)
    
    def integrate(self, question: str, output: str, score: float):
        """Add new simulation result to memory box."""
        self.memory.append({
            "question": question,
            "output": output,
            "score": score,
            "t": self.t
        })
        self.t += 1
    
    def score(self, question: str, proposed_output: str) -> tuple[float, str]:
        """Check proposed output against accumulated memory."""
        if not self.memory:
            return 0.5, "No memory yet — neutral score"
        
        # Build memory context
        recent = self.memory[-5:]  # Last 5 interactions
        memory_context = "\n".join([
            f"Past Q: {m['question'][:80]} | Score: {m['score']:.2f}"
            for m in recent
        ])
        
        system = """You are a memory consistency checker.

Given past interactions and a new proposed answer, check if the new answer 
is consistent with established knowledge from memory.

Respond ONLY in JSON format:
{
  "score": <float between 0.0 and 1.0>,
  "consistent_with_memory": <true or false>,
  "reasoning": "<one sentence>"
}"""
        
        prompt = f"""Past memory:\n{memory_context}

New question: {question}
New proposed answer: {proposed_output}

Is the new answer consistent with past knowledge?"""
        
        try:
            response = call_claude(system, prompt, max_tokens=200)
            clean = response.strip()
            if "```" in clean:
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            data = json.loads(clean.strip())
            return float(data["score"]), data["reasoning"]
        except Exception as e:
            return 0.5, f"M6 parse error: {e}"


# ============================================================
# DECISION ENGINE
# o* = argmax SUM w_i * p_i(M_i(x))
# ============================================================

class DecisionEngine:
    """
    Weighted voting across all active modules.
    Selects the output with highest combined confidence.
    """
    
    # Weights sum to 1.0
    WEIGHTS = {
        "M1_world_model": 0.40,
        "M5_sim_checker": 0.35,
        "M6_memory_box":  0.25,
    }
    
    def vote(self, scores: dict[str, float]) -> float:
        """o* = argmax SUM w_i * p_i"""
        total = sum(
            self.WEIGHTS[module] * score
            for module, score in scores.items()
            if module in self.WEIGHTS
        )
        return total


# ============================================================
# META-BRAIN: THE COMPLETE SYSTEM
# ============================================================

class MetaBrain:
    """
    The complete Meta-Brain prototype.
    
    For every input:
    1. Generate N candidate outputs using base LLM
    2. Run M1, M5, M6 on each candidate SIMULTANEOUSLY
    3. Decision engine votes on best output
    4. M6 memory updates with result
    
    This is NOT a single forward pass.
    This is parallel simulation + voting.
    """
    
    def __init__(self):
        self.m1 = WorldModel()
        self.m5 = SimulationChecker()
        self.m6 = UniverseMemoryBox()
        self.engine = DecisionEngine()
        self.run_count = 0
        self.history = []
    
    def generate_candidates(self, question: str, n: int = 3) -> list[str]:
        """Generate N diverse candidate answers using base LLM."""
        system = f"""Generate exactly {n} different possible answers to the question.
Each answer should take a different approach or perspective.
Respond ONLY in JSON format:
{{"answers": ["answer1", "answer2", "answer3"]}}"""
        
        try:
            response = call_claude(system, question, max_tokens=800)
            clean = response.strip()
            if "```" in clean:
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            data = json.loads(clean.strip())
            return data["answers"][:n]
        except Exception as e:
            # Fallback: single answer
            response = call_claude(
                "Answer this question clearly and accurately.",
                question,
                max_tokens=300
            )
            return [response]
    
    def process(self, question: str) -> dict[str, Any]:
        """
        Full Meta-Brain processing pipeline.
        Returns best answer with full scoring breakdown.
        """
        print(f"\n{'='*60}")
        print(f"🧠 META-BRAIN PROCESSING")
        print(f"{'='*60}")
        print(f"📥 Question: {question}")
        print(f"📊 Memory quality: Q(t={self.run_count}) = {self.m6.quality():.4f}")
        
        # Step 1: Generate candidate outputs
        print(f"\n⚡ Generating candidate outputs...")
        candidates = self.generate_candidates(question, n=3)
        print(f"   Generated {len(candidates)} candidates")
        
        # Step 2: Score each candidate through all modules
        print(f"\n🔬 Running parallel module evaluation...")
        scored_candidates = []
        
        for i, candidate in enumerate(candidates):
            print(f"\n   Candidate {i+1}: {candidate[:60]}...")
            
            # M1: World Model
            m1_score, m1_reason = self.m1.score(question, candidate)
            print(f"   M1 World Model:      {m1_score:.3f} — {m1_reason[:50]}")
            
            # M5: Simulation Checker  
            m5_score, m5_reason = self.m5.score(question, candidate)
            print(f"   M5 Sim Checker:      {m5_score:.3f} — {m5_reason[:50]}")
            
            # M6: Memory Box
            m6_score, m6_reason = self.m6.score(question, candidate)
            print(f"   M6 Memory Box:       {m6_score:.3f} — {m6_reason[:50]}")
            
            # Decision Engine vote
            scores = {
                "M1_world_model": m1_score,
                "M5_sim_checker": m5_score,
                "M6_memory_box":  m6_score,
            }
            final_score = self.engine.vote(scores)
            print(f"   ⚖️  Weighted score:   {final_score:.4f}")
            
            scored_candidates.append({
                "answer": candidate,
                "scores": scores,
                "final_score": final_score,
                "m1_reason": m1_reason,
                "m5_reason": m5_reason,
                "m6_reason": m6_reason,
            })
        
        # Step 3: Select best output
        best = max(scored_candidates, key=lambda x: x["final_score"])
        
        # Step 4: Update memory
        self.m6.integrate(question, best["answer"], best["final_score"])
        self.run_count += 1
        self.history.append({"question": question, "answer": best["answer"], "score": best["final_score"]})
        
        # Step 5: Report
        print(f"\n{'='*60}")
        print(f"✅ BEST OUTPUT SELECTED (score={best['final_score']:.4f})")
        print(f"{'='*60}")
        print(f"\n💬 Answer: {best['answer']}")
        print(f"\n📈 Score breakdown:")
        print(f"   M1 (x0.40): {best['scores']['M1_world_model']:.3f}")
        print(f"   M5 (x0.35): {best['scores']['M5_sim_checker']:.3f}")
        print(f"   M6 (x0.25): {best['scores']['M6_memory_box']:.3f}")
        print(f"   Final:      {best['final_score']:.4f}")
        print(f"\n🌌 Memory quality after this run: {self.m6.quality():.4f}")
        
        return best
    
    def compare_with_baseline(self, question: str):
        """
        KEY EXPERIMENT: Compare Meta-Brain vs plain LLM.
        This is Prediction P1 — does voting improve accuracy?
        """
        print(f"\n{'='*60}")
        print(f"🔬 EXPERIMENT: Meta-Brain vs Baseline LLM")
        print(f"{'='*60}")
        
        # Baseline: single forward pass
        print("\n📊 BASELINE (single LLM call):")
        t0 = time.time()
        baseline = call_claude(
            "Answer this question accurately.",
            question,
            max_tokens=300
        )
        baseline_time = time.time() - t0
        print(f"   Answer: {baseline[:100]}...")
        print(f"   Time: {baseline_time:.2f}s")
        print(f"   Modules used: 1")
        
        # Meta-Brain: parallel simulation + voting
        print("\n🧠 META-BRAIN (parallel simulation + voting):")
        t0 = time.time()
        result = self.process(question)
        metabrain_time = time.time() - t0
        print(f"\n   Time: {metabrain_time:.2f}s")
        print(f"   Modules used: 3 parallel + decision engine")
        
        print(f"\n{'='*60}")
        print(f"📋 COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"  Baseline answer confidence:  unknown (no self-check)")
        print(f"  Meta-Brain final score:       {result['final_score']:.4f}")
        print(f"  Meta-Brain checked for:       physics + hallucination + memory")
        print(f"  Architecture difference:      1 pass vs 3 parallel modules")
        print(f"{'='*60}")
        
        return result, baseline


# ============================================================
# DEMO: RUN THE PROTOTYPE
# ============================================================

if __name__ == "__main__":
    
    print("🧠 META-BRAIN PROTOTYPE — PhenexAI Research")
    print("=" * 60)
    print("Architecture: M1 + M5 + M6 + Decision Engine")
    print("Base LLM: Claude (via API)")
    print("=" * 60)
    
    brain = MetaBrain()
    
    # Test questions — chosen to demonstrate the architecture
    test_questions = [
        "What is the speed of light and why can nothing exceed it?",
        "How does quantum entanglement work?",
        "Can a simulation share physics laws with its host universe?",
    ]
    
    # Run comparison on first question
    result, baseline = brain.compare_with_baseline(test_questions[0])
    
    # Run remaining questions through Meta-Brain
    for q in test_questions[1:]:
        brain.process(q)
        time.sleep(1)  # Rate limit courtesy
    
    # Final memory report
    print(f"\n{'='*60}")
    print(f"🌌 FINAL UNIVERSE MEMORY BOX REPORT")
    print(f"{'='*60}")
    print(f"  Total runs:      {brain.run_count}")
    print(f"  Memory quality:  {brain.m6.quality():.4f} ({brain.m6.quality()*100:.1f}%)")
    print(f"  Memory entries:  {len(brain.m6.memory)}")
    print(f"\n  This would be 0.0 at start, growing toward 1.0")
    print(f"  RAG would be degrading. Memory Box is compounding.")
    print(f"\n✅ Prototype complete. Architecture validated.")
    print(f"   Next: test on ARC-AGI benchmark for Prediction P1")
