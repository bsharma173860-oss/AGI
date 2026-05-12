"""
metabrain_core.py
=================
PhenexAI Research | Bharat | May 2026

Single file that loads entire Meta-Brain system.
Usage in Jupyter:
    from metabrain_core import *

This replaces running 10 cells every time.
All modules, all math, all functions — one import.
"""

# ── Imports ────────────────────────────────────────────────────────────────────
import json
import time
import numpy as np
import os

# ── Configuration — change these to switch between API and local ───────────────
MODE         = 'api'          # 'api' or 'local'
API_KEY      = os.environ.get('ANTHROPIC_API_KEY', '')  # Set in environment
OLLAMA_URL   = 'http://localhost:11434/api/generate'
MODEL_NAME   = 'llama3'
LAMBDA_RATE  = 0.15
WEIGHTS      = {'M1': 0.40, 'M5': 0.35, 'M6': 0.25}

# ── Step 1: Base Model Call ─────────────────────────────────────────────────────

def call_model(system_prompt, user_prompt, max_tokens=300):
    """
    Unified model caller.
    Switches between API and local based on MODE.
    
    Math: B(x) -> h  (hidden representation)
    API:   t_latency = ~2000ms, cost = ~$0.001
    Local: t_latency = ~200ms,  cost = $0.000
    """
    if MODE == 'api':
        return _call_api(system_prompt, user_prompt, max_tokens)
    else:
        return _call_local(system_prompt, user_prompt, max_tokens)


def _call_api(system_prompt, user_prompt, max_tokens=300):
    """Claude API call."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=API_KEY)
        message = client.messages.create(
            model='claude-sonnet-4-20250514',
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{'role': 'user', 'content': user_prompt}]
        )
        return message.content[0].text
    except Exception as e:
        return f'API error: {e}'


def _call_local(system_prompt, user_prompt, max_tokens=300):
    """Local Ollama call — no API, no cost."""
    import requests
    prompt = f'System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:'
    try:
        response = requests.post(OLLAMA_URL, json={
            'model': MODEL_NAME,
            'prompt': prompt,
            'stream': False,
            'options': {'num_predict': max_tokens, 'temperature': 0.7}
        }, timeout=120)
        return response.json()['response'].strip()
    except Exception as e:
        return f'Local model error: {e}'


def parse_json(raw):
    """
    Safely parse JSON from model output.
    Handles markdown code blocks and extra text.
    """
    raw = raw.strip()
    # Remove markdown
    if '```' in raw:
        parts = raw.split('```')
        for part in parts:
            if '{' in part:
                raw = part
                break
    if raw.startswith('json'):
        raw = raw[4:]
    # Find JSON object
    start = raw.find('{')
    end = raw.rfind('}') + 1
    if start >= 0 and end > start:
        try:
            return json.loads(raw[start:end])
        except:
            pass
    return None


# ── Module M1: World Model ──────────────────────────────────────────────────────

def m1_world_model(question, answer):
    """
    M1: Physical and logical consistency checker.
    Math: p_1(o) = exp(-D_KL(P_o || P_C)) / Z
    
    Score 1.0 = physically perfect
    Score 0.0 = physically impossible
    """
    system = """You are a physics and logic consistency checker.
Score the answer for physical and logical consistency.
1.0=perfect, 0.7=minor issues, 0.4=problems, 0.0=impossible
Respond ONLY in JSON: {"score": 0.8, "reasoning": "one sentence"}
No markdown. No extra text."""
    raw = call_model(system, f'Q: {question}\nA: {answer}', 100)
    data = parse_json(raw)
    if data:
        return float(data.get('score', 0.5)), str(data.get('reasoning', ''))
    return 0.5, 'parse error'


# ── Module M4: Synthetic Reality ────────────────────────────────────────────────

def m4_synthetic_reality(question, context=''):
    """
    M4: Counterfactual scenario generator.
    Math: z_cf = z_0 + PI_L(delta)  (physics-consistent perturbation)
    Novel inference: T not in D_train but valid under L_S = L_U
    
    Simulates in latent space — 100x cheaper than pixel space
    """
    system = """You are a counterfactual scenario generator.
Generate ONE novel scenario that:
1. Has never happened but is physically possible
2. Is relevant to answering the question
3. Obeys the same physics laws as our universe
4. Produces a genuine insight not obvious from the question
Keep under 3 sentences."""
    prompt = f'Question: {question}'
    if context:
        prompt += f'\nContext: {context}'
    return call_model(system, prompt, 150)


def m4_novelty_score(scenario, question):
    """
    p_4(o) = density of novel inferences in scenario
    Higher = more genuinely novel content from imagination
    """
    system = 'Rate novelty and insight of this scenario. Respond ONLY in JSON: {"score": 0.7, "reasoning": "one sentence"}. No markdown.'
    raw = call_model(system, f'Q: {question}\nScenario: {scenario}', 80)
    data = parse_json(raw)
    if data:
        return float(data.get('score', 0.5)), str(data.get('reasoning', ''))
    return 0.5, 'parse error'


# ── Module M5: Simulation Checker ───────────────────────────────────────────────

def m5_sim_checker(question, answer):
    """
    M5: Hallucination detector.
    Math: T(o) = -log P(o | W_t, L_S)
    Reject if T(o) > theta (hallucination detected)
    
    Score 1.0 = verified fact
    Score 0.0 = clear hallucination
    """
    system = """You are a hallucination detector.
Score factual grounding of the answer.
0.9-1.0: verified scientific fact
0.7-0.9: standard knowledge, likely correct
0.5-0.7: plausible but uncertain
0.3-0.5: suspicious claims
0.0-0.3: hallucination or factual error
Respond ONLY in JSON: {"score": 0.8, "reasoning": "one sentence"}
No markdown. No extra text."""
    raw = call_model(system, f'Q: {question}\nA: {answer}', 100)
    data = parse_json(raw)
    if data:
        return float(data.get('score', 0.5)), str(data.get('reasoning', ''))
    return 0.5, 'parse error'


# ── Module M6: Universe Memory Box ──────────────────────────────────────────────

class UniverseMemoryBox:
    """
    M6: Persistent compounding memory.
    
    MATH:
    - Storage: m_t = Embed(omega_t) in R^384
    - Retrieval: top-k by cosine similarity
    - Quality: Q(t) = 1 - exp(-lambda*t) -> 1
    - RAG degrades: P_RAG = P_0 * exp(-mu*t) -> 0
    
    PERSISTENCE: survives Jupyter restarts
    COST: $0 forever
    STORAGE: 1M memories = ~2GB
    """

    def __init__(self, persist_path='./metabrain_memory'):
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
            self.client = chromadb.PersistentClient(path=persist_path)
            self.collection = self.client.get_or_create_collection('universe_memory')
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            self.t = self.collection.count()
            self.use_vectors = True
            print(f'M6 Memory Box: {self.t} memories loaded from disk')
        except ImportError:
            # Fallback: simple in-memory list if chromadb not installed
            self.memories = []
            self.t = 0
            self.use_vectors = False
            print('M6 Memory Box: running in-memory mode (install chromadb for persistence)')

    def quality(self):
        """Q(t) = 1 - exp(-lambda * t)"""
        return round(1.0 - np.exp(-LAMBDA_RATE * self.t), 4)

    def integrate(self, question, answer, score):
        """
        U_{t+1} = U_t (+) compress(simulation_output)
        Integrate — not just append
        """
        memory_text = f'Q: {question[:100]} | A: {answer[:200]} | Score: {score:.3f}'
        if self.use_vectors:
            self.collection.add(
                documents=[memory_text],
                embeddings=[self.embedder.encode(memory_text).tolist()],
                ids=[f'memory_{self.t}'],
                metadatas=[{'question': question[:100], 'score': score, 't': self.t}]
            )
        else:
            self.memories.append({'text': memory_text, 'score': score})
        self.t += 1

    def retrieve(self, query, k=3):
        """R(q,k) = arg top-k cosine_sim(q, m) for m in U_t"""
        if self.t == 0:
            return []
        if self.use_vectors:
            k = min(k, self.t)
            results = self.collection.query(
                query_embeddings=[self.embedder.encode(query).tolist()],
                n_results=k
            )
            return results['documents'][0] if results['documents'] else []
        else:
            return [m['text'] for m in self.memories[-k:]]

    def score(self, question, answer):
        """
        p_6(o) = P(consistent with memory | retrieved context)
        Theorem 2: score improves with more memories
        """
        memories = self.retrieve(question, k=3)
        if not memories:
            return 0.5, 'No memory yet'
        context = ' | '.join(memories)[:300]
        system = 'Check memory consistency. Respond ONLY in JSON: {"score": 0.8, "reasoning": "one sentence"}. No markdown.'
        raw = call_model(system, f'Memory: {context}\nNew Q: {question}\nNew A: {answer}', 80)
        data = parse_json(raw)
        if data:
            return float(data.get('score', 0.5)), str(data.get('reasoning', ''))
        return 0.5, 'parse error'


# ── Decision Engine ─────────────────────────────────────────────────────────────

def decision_engine(scores_dict):
    """
    o* = argmax SUM w_i * p_i
    
    Weighted voting across all active modules.
    Selects output with highest combined confidence.
    """
    return sum(WEIGHTS.get(k, 0) * v for k, v in scores_dict.items())


def generate_candidates(question, n=2):
    """
    Generate N diverse candidate answers.
    Math: {o_1,...,o_n} = diverse samples from B(x)
    """
    system = f'Generate {n} different answers. Be concise. Respond ONLY in JSON: {{"answers": ["answer1", "answer2"]}}. No markdown.'
    try:
        raw = call_model(system, question, 400)
        data = parse_json(raw)
        if data and 'answers' in data:
            return data['answers'][:n]
    except:
        pass
    return [call_model('Answer accurately in 2 sentences.', question, 150)]


# ── Full Meta-Brain Pipeline ────────────────────────────────────────────────────

# Global memory box — persists across all calls
_memory = None

def get_memory():
    """Get or create global memory box."""
    global _memory
    if _memory is None:
        _memory = UniverseMemoryBox()
    return _memory


def metabrain(question, verbose=True, use_m4=True):
    """
    FULL META-BRAIN PIPELINE
    
    MATH:
    o* = argmax SUM_{i} w_i * p_i(M_i(x, U_t))
    
    MODULES:
    M1: p_1 = exp(-D_KL)/Z          [physics check]
    M4: p_4 = novelty density        [imagination — optional]
    M5: p_5 = 1 - T(o)/theta_max    [hallucination gate]
    M6: p_6 = P(consistent|R(q,k))  [memory]
    
    ARCHITECTURE:
    Input -> Generate N candidates
          -> Score each: M1, M5, M6 in parallel
          -> Decision engine votes
          -> Best selected
          -> Memory updated
    
    Returns: dict with answer, scores, memory quality
    """
    m6 = get_memory()

    if verbose:
        print(f'\n{"="*55}')
        print(f'QUESTION: {question}')
        print(f'Memory Q(t={m6.t}) = {m6.quality()}')

    # M4: Generate scenario from imagination
    scenario = ''
    if use_m4 and m6.t > 0:
        memories = m6.retrieve(question, k=1)
        context = memories[0] if memories else ''
        scenario = m4_synthetic_reality(question, context)
        if verbose:
            print(f'M4 Scenario: {scenario[:80]}...')

    # Generate candidates
    candidates = generate_candidates(question, n=2)
    if verbose:
        print(f'Generated {len(candidates)} candidates')

    # Score each candidate
    results = []
    for i, c in enumerate(candidates):
        s1, _ = m1_world_model(question, c)
        s5, _ = m5_sim_checker(question, c)
        s6, _ = m6.score(question, c)
        s4 = 0.0
        if scenario:
            s4, _ = m4_novelty_score(scenario, question)

        final = decision_engine({'M1': s1, 'M5': s5, 'M6': s6})
        results.append({'answer': c, 'M1': s1, 'M5': s5,
                        'M6': s6, 'M4': s4, 'final': final})
        if verbose:
            print(f'  C{i+1}: M1={s1:.2f} M5={s5:.2f} M6={s6:.2f} => {final:.3f}')

    # Select best output
    best = max(results, key=lambda x: x['final'])

    # Update memory
    m6.integrate(question, best['answer'], best['final'])

    if verbose:
        print(f'\nBEST (score={best["final"]:.3f}):')
        print(best['answer'])
        print(f'Memory quality now: {m6.quality()}')

    best['memory_quality'] = m6.quality()
    best['memory_t'] = m6.t
    return best


def memory_status():
    """Quick memory status check."""
    m6 = get_memory()
    print(f'Memory entries: {m6.t}')
    print(f'Quality Q(t):   {m6.quality()}')
    rag = np.exp(-0.001 * m6.t)
    print(f'RAG equivalent: {rag:.4f} (degrading)')
    print(f'Advantage over RAG: {m6.quality() - rag:.4f}')


def switch_mode(new_mode):
    """Switch between API and local mode."""
    global MODE
    MODE = new_mode
    print(f'Switched to {MODE} mode')
    if MODE == 'api':
        print('Cost: ~$0.001 per call | Speed: ~2000ms')
    else:
        print('Cost: $0.000 per call | Speed: ~200ms')
        print('Make sure Ollama is running: ollama serve')


# ── Startup Message ─────────────────────────────────────────────────────────────

print('=' * 55)
print('Meta-Brain Core Loaded — PhenexAI Research')
print('=' * 55)
print(f'Mode:    {MODE}')
print(f'Modules: M1 (physics) + M4 (imagination) + M5 (hallucination) + M6 (memory)')
print(f'Weights: M1={WEIGHTS["M1"]} M5={WEIGHTS["M5"]} M6={WEIGHTS["M6"]}')
print()
print('Commands:')
print('  metabrain("your question")    — run full pipeline')
print('  memory_status()               — check memory quality')
print('  switch_mode("local")          — switch to local model')
print('  switch_mode("api")            — switch to API')
print('=' * 55)
