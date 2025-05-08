from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from pdf2image import convert_from_path
import pytesseract
import requests
from bs4 import BeautifulSoup
from fpdf import FPDF
import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
load_dotenv()


# === Graph State ===
class GraphState(TypedDict):
    topic: str | None
    pdf_path: str | None
    text: str | None
    summary: str | None
    bullet_points: list[str] | None
    mcqs: list[str] | None
    google_links: list[str] | None 
    book_recommendations: list[str] | None 

# === Wikipedia Fetch ===
def fetch_wikipedia_summary(topic: str) -> str:
    url = f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}"
    response = requests.get(url)
    if response.status_code != 200:
        return f"Could not fetch Wikipedia page for: {topic}"
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.select("p")
    content = " ".join(p.get_text() for p in paragraphs[:5])
    return content.strip()

# === GeeksforGeeks Fetch ===
def fetch_gfg_summary(topic: str) -> str:
    search_query = topic.replace(" ", "+") + "+site:geeksforgeeks.org"
    headers = {"User-Agent": "Mozilla/5.0"}
    search_url = f"https://www.google.com/search?q={search_query}"

    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    links = [a['href'] for a in soup.select("a") if "geeksforgeeks.org" in a.get('href', '')]

    for link in links:
        try:
            real_url = link.split("&")[0].replace("/url?q=", "")
            article = requests.get(real_url, headers=headers)
            soup = BeautifulSoup(article.text, "html.parser")
            paragraphs = soup.select("div.content p") or soup.select("article p")
            content = " ".join(p.get_text() for p in paragraphs[:5])
            if content:
                return content.strip()
        except:
            continue
    return "No relevant content found from GeeksforGeeks."

# === Google Search Links ===
def get_google_links(topic: str) -> list[str]:
    search_url = f"https://www.google.com/search?q={topic.replace(' ', '+')}"
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(res.text, "html.parser")
    anchors = soup.select("a")
    links = []
    for a in anchors:
        href = a.get("href", "")
        if href.startswith("/url?q="):
            clean_link = href.split("&")[0].replace("/url?q=", "")
            if "http" in clean_link and "google" not in clean_link:
                links.append(clean_link)
        if len(links) >= 5:
            break
    return links

# === Combined Web Search Node ===
def web_search_node(state: GraphState) -> dict:
    topic = state["topic"]
    wiki = fetch_wikipedia_summary(topic)
    gfg = fetch_gfg_summary(topic)
    google_urls = get_google_links(topic)

    combined_text = wiki + "\n\n" + gfg
    return {
        "text": combined_text,
        "google_links": google_urls
    }

# === PDF Loader Node ===
def extract_text_from_pdf(file_path: str) -> str:
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        return "\n".join(page.page_content for page in pages)
    except Exception:
        pass  # fallback to OCR
    text = ""
    images = convert_from_path(file_path)
    for image in images:
        text += pytesseract.image_to_string(image)
    return text

def pdf_loader_node(state: GraphState) -> dict:
    text = extract_text_from_pdf(state["pdf_path"])
    return {"text": text}

# === Flow Controller ===
def input_router(state: GraphState) -> Annotated[str, {"next": str}]:
    if state.get("pdf_path"):
        return {"next": "load_pdf"}
    elif state.get("topic"):
        return {"next": "search_web"}
    else:
        raise ValueError("Input must contain either 'topic' or 'pdf_path'.")

# === OpenAI Model ===
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key: 
    raise ValueError("Please set the environment variable.")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, openai_api_key=openai_api_key)

# === Summarizer Node ===
def summarize_text(text: str) -> str:
    prompt = f"""Summarize the following text in about 300 words for a government exam aspirant. make a combined summary from all sources . Use clear and factual language:\n\n{text}"""
    return llm.invoke(prompt).content

def summarizer_node(state: GraphState) -> dict:
    summary = summarize_text(state["text"])
    return {"summary": summary}

# === Bullet Points Node ===
def extract_bullet_points(text: str) -> list[str]:
    prompt = f"""Extract 15 critically important bullet points from the following educational content for a government exam aspirant. Keep them precise and factual.\n\n{text}\n\nReturn as a numbered list."""
    result = llm.invoke(prompt).content
    return result.strip().split("\n")

def bullet_node(state: GraphState) -> dict:
    bullet_points = extract_bullet_points(state["text"])
    return {"bullet_points": bullet_points}

# === MCQ Node ===
def generate_mcqs(text: str) -> list[str]:
    prompt = f"""Create 15 multiple choice questions (MCQs) from the following content with 4 options for each question. At the end, give a list of correct answers using question number and answer option in captal letters.\n\n
    {text}"""
    result = llm.invoke(prompt).content
    return result.strip().split('\n\n')

def mcq_node(state: GraphState) -> dict:
    mcqs = generate_mcqs(state["text"])
    return {"mcqs": mcqs}


def recommend_books(text : str) -> list[str]:
    prompt = f"""Recommend 5 related to topic books that would be helpful for a government exam aspirant. Provide the title and author for each book:\n\n
    {text}"""
    result = llm.invoke(prompt).content
    return result.strip().split('\n')

def book_node(state: GraphState) -> dict:
    books = recommend_books(state["text"])
    return {"book_recommendations": books}


# === LangGraph Flow ===
graph = StateGraph(GraphState)
graph.add_node("input_route", input_router)
graph.add_node("search_web", web_search_node)
graph.add_node("load_pdf", pdf_loader_node)
graph.add_node("summarizer", summarizer_node)
graph.add_node("15_bullet_points", bullet_node)
graph.add_node("15_mcqs", mcq_node)
graph.add_node("book_recommendation", book_node)

graph.add_conditional_edges(
    "input_route",
    lambda x: x["next"],
    {
        "load_pdf": "load_pdf",
        "search_web": "search_web"
    }
)

graph.set_entry_point("input_route")
graph.add_edge("search_web", "summarizer")
graph.add_edge("load_pdf", "summarizer")
graph.add_edge("summarizer", "15_bullet_points")
graph.add_edge("15_bullet_points", "15_mcqs")
graph.add_edge("15_mcqs", "book_recommendation")
graph.add_edge("book_recommendation", END)

app = graph.compile()

# === Output Formatter ===
def format_output(result):
    output = "=== SUMMARY ===\n"
    output += result.get("summary", "No summary available") + "\n\n"
    
    output += "=== BULLET POINTS ===\n"
    bullet_points = result.get("bullet_points", [])
    if bullet_points:
        output += "\n".join(bullet_points) + "\n\n"
    else:
        output += "No bullet points available\n\n"
    
    output += "=== MCQs ===\n"
    mcqs = result.get("mcqs", [])
    if mcqs:
        output += "\n\n".join(mcqs) + "\n\n"
    else:
        output += "No MCQs available\n\n"

    output += "=== BOOK RECOMMENDATIONS ===\n"
    books = result.get("book_recommendations", [])
    if books:
        output += "\n".join(books) + "\n\n"
    else:
        output += "No book recommendations available\n\n"    

    output += "=== GOOGLE LINKS ===\n"
    google_links = result.get("google_links", [])
    if google_links:
        output += "\n".join(google_links)
    else:
        output += "No links found."

    return output

from fpdf import FPDF

def save_pdf(content: str, filename: str = "output.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Add a Unicode-capable font
    pdf.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
    pdf.set_font('DejaVu', '', 12)

    for line in content.split('\n'):
        pdf.multi_cell(0, 10, line)

    pdf.output(filename)


# === Test Run ===
# result_1 = app.invoke({"topic": "Indian Constitution"})
# print(format_output(result_1))
