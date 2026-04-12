from dotenv import load_dotenv
load_dotenv()

import operator
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from ddgs import DDGS

llm = ChatOllama(model="mistral:latest", temperature=0.3)

# ─── State ────────────────────────────────────────────────────────────────────
class ArticleState(TypedDict):
    topic:         str
    outline:       str
    research:      Annotated[list[str], operator.add]
    draft:         str
    final_article: str
    next_agent:    str   # supervisor bunu yönetir
    step_count:    int

# ─── Yardımcı: web arama ──────────────────────────────────────────────────────
def web_search(query: str, max_results: int = 4) -> list[str]:
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results, timeout=8):
                results.append(f"{r['title']}: {r['body']}")
    except Exception as e:
        print(f"   Arama hatasi: {e}")
    return results

# ─── SUPERVISOR ───────────────────────────────────────────────────────────────
def supervisor_node(state: ArticleState) -> dict:
    """
    Hangi agent'in çalışacağına karar verir.
    Sıra: planner → researcher → writer → editor → END
    """
    step = state.get("step_count", 0)

    if not state.get("outline"):
        next_a = "planner"
    elif not state.get("research"):
        next_a = "researcher"
    elif not state.get("draft"):
        next_a = "writer"
    elif not state.get("final_article"):
        next_a = "editor"
    else:
        next_a = "end"

    print(f"\n🎯  [supervisor] Adim {step+1} → {next_a}")
    return {"next_agent": next_a, "step_count": step + 1}

def supervisor_router(state: ArticleState) -> str:
    return state["next_agent"]

# ─── PLANNER AGENT ────────────────────────────────────────────────────────────
def planner_agent(state: ArticleState) -> dict:
    print("📋  [planner_agent] Outline hazırlanıyor...")

    response = llm.invoke([
        SystemMessage(content="""Sen bir makale planlayıcısısın.
Verilen konuya göre 5 bölümlük makale outline'ı çıkar.
Format:
1. Giriş
2. [Bölüm adı]
3. [Bölüm adı]
4. [Bölüm adı]
5. Sonuç
Sadece outline'ı ver, başka açıklama yapma."""),
        HumanMessage(content=f"Makale konusu: {state['topic']}")
    ])

    print(f"   Outline hazır:\n{response.content}\n")
    return {"outline": response.content}

# ─── RESEARCHER AGENT ─────────────────────────────────────────────────────────
def researcher_agent(state: ArticleState) -> dict:
    print("🔍  [researcher_agent] Araştırma yapılıyor...")

    # Outline'dan 3 arama sorgusu üret
    response = llm.invoke([
        SystemMessage(content="""Bu outline için 3 adet kısa İngilizce arama sorgusu üret.
Her sorguyu yeni satıra yaz. Sadece sorguları ver."""),
        HumanMessage(content=f"Konu: {state['topic']}\nOutline:\n{state['outline']}")
    ])

    queries = [q.strip() for q in response.content.strip().split("\n") if q.strip()][:3]
    all_results = []

    for q in queries:
        print(f"   Aranıyor: {q[:50]}")
        hits = web_search(q, max_results=3)
        all_results.extend(hits)

    print(f"   {len(all_results)} kaynak bulundu\n")
    return {"research": all_results}

# ─── WRITER AGENT ─────────────────────────────────────────────────────────────
def writer_agent(state: ArticleState) -> dict:
    print("✍️   [writer_agent] Makale yazılıyor...")

    research_text = "\n".join(state["research"][:10])

    response = llm.invoke([
        SystemMessage(content="""Sen bir makale yazarısın. Türkçe yaz.
Verilen outline ve araştırma materyallerini kullanarak
akıcı, bilgilendirici bir makale yaz.
Her bölümü 2-3 paragraf yaz.
Markdown format kullan (## başlıklar)."""),
        HumanMessage(content=f"""
Konu: {state['topic']}

Outline:
{state['outline']}

Araştırma Materyalleri:
{research_text}

Makaleyi yaz:
""")
    ])

    print(f"   Taslak hazır ({len(response.content)} karakter)\n")
    return {"draft": response.content}

# ─── EDITOR AGENT ─────────────────────────────────────────────────────────────
def editor_agent(state: ArticleState) -> dict:
    print("✏️   [editor_agent] Makale düzenleniyor...")

    response = llm.invoke([
        SystemMessage(content="""Sen bir editörsün. Türkçe yaz.
Verilen makale taslağını şu kriterlere göre düzenle:
- Akıcılık ve okunabilirlik artır
- Tekrar eden ifadeleri kaldır
- Giriş ve sonucu güçlendir
- Varsa hatalı bilgileri düzelt
- Markdown formatını koru
Düzenlenmiş son halini ver."""),
        HumanMessage(content=f"Makale taslağı:\n{state['draft']}")
    ])

    print(f"   Final makale hazır ({len(response.content)} karakter)\n")
    return {"final_article": response.content}

# ─── GRAPH ────────────────────────────────────────────────────────────────────
def build_graph():
    graph = StateGraph(ArticleState)

    # Node'ları ekle
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("planner",    planner_agent)
    graph.add_node("researcher", researcher_agent)
    graph.add_node("writer",     writer_agent)
    graph.add_node("editor",     editor_agent)

    # Entry point
    graph.set_entry_point("supervisor")

    # Supervisor → routing
    graph.add_conditional_edges(
        "supervisor",
        supervisor_router,
        {
            "planner":    "planner",
            "researcher": "researcher",
            "writer":     "writer",
            "editor":     "editor",
            "end":        END
        }
    )

    # Her agent bittikten sonra supervisor'a döner
    graph.add_edge("planner",    "supervisor")
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("writer",     "supervisor")
    graph.add_edge("editor",     "supervisor")

    return graph.compile()

# ─── ÇALIŞTIR ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time

    app = build_graph()

    # Graph PNG kaydet
    try:
        with open("article_graph.png", "wb") as f:
            f.write(app.get_graph().draw_mermaid_png())
        print("Graph kaydedildi: article_graph.png\n")
    except Exception:
        pass

    konu = input("Makale konusu → ")

    print(f"\n{'='*50}")
    print(f"Konu: {konu}")
    print(f"{'='*50}\n")

    start = time.time()

    result = app.invoke({
        "topic":         konu,
        "outline":       "",
        "research":      [],
        "draft":         "",
        "final_article": "",
        "next_agent":    "",
        "step_count":    0
    })

    sure = time.time() - start

    print("\n" + "="*50)
    print("FINAL MAKALE:")
    print("="*50)
    print(result["final_article"])
    print(f"\nToplam süre: {sure:.1f} saniye")

    # Dosyaya kaydet
    with open("makale.md", "w", encoding="utf-8") as f:
        f.write(f"# {konu}\n\n")
        f.write(result["final_article"])
    print("Makale kaydedildi: makale.md")