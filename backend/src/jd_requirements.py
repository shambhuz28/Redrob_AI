import os
import json
import re
import unicodedata
from pathlib import Path
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

from dotenv import load_dotenv
from google import genai

import pymupdf

def jd(jd_path):
    suffix = Path(jd_path).suffix.lower()

    if suffix == ".docx":
        return read_docx(jd_path)

    info = ""
    doc = pymupdf.open(jd_path)

    for page in doc:
        info += page.get_text()

    return info


def read_docx(jd_path):
    try:
        with ZipFile(jd_path) as archive:
            document_xml = archive.read("word/document.xml")
    except (KeyError, BadZipFile) as exc:
        raise ValueError("Could not read DOCX job description text.") from exc

    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    root = ElementTree.fromstring(document_xml)
    paragraphs = []

    for paragraph in root.findall(".//w:p", namespace):
        text = "".join(
            node.text or ""
            for node in paragraph.findall(".//w:t", namespace)
        ).strip()
        if text:
            paragraphs.append(text)

    return "\n".join(paragraphs)
    
def clean_llm_output(text: str) -> str:
    """Clean LLM output before parsing JSON."""

    if not text:
        return ""

    text = text.strip()

    # Remove markdown code fences if present
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```$", "", text)

    # Normalize unicode
    text = unicodedata.normalize("NFKD", text)

    # Replace fancy punctuation
    text = re.sub(r"[‐-–—]", "-", text)
    text = re.sub(r"[‘’]", "'", text)
    text = re.sub(r'[“”]', '"', text)

    return text.strip()


def req_from_jd(jd_path):

    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env")

    client = genai.Client(api_key=api_key)

    
    job_description = jd(jd_path)

    prompt = """
You are an expert at converting Job Descriptions into semantic search queries for resume retrieval.

Your response will be embedded into vector embeddings.

Return ONLY valid JSON.

Schema:

{
    "career": "...",
    "skills": "...",
    "summary": "..."
}

==========================
RULES
==========================

1. career

Extract:

- Responsibilities
- Expected work experience
- Projects
- Domains worked in
- Years of experience
- Technologies used while working

Do NOT include company information.

----------------------------------------

2. skills

Extract ALL technical skills.

DO NOT remove any buzzwords.

Examples:

RAG
LoRA
LangChain
LlamaIndex
Qdrant
Milvus
Pinecone
FAISS
Sentence Transformers
Transformers
BGE
OpenAI
Claude
Gemini
PyTorch
TensorFlow
CUDA
PEFT
FastAPI
Docker
Kubernetes

Whenever a buzzword appears:

KEEP IT.

Then immediately expand it with concise semantic phrases.

Example:

RAG

Retrieval-Augmented Generation

retrieving relevant documents

semantic search

vector databases

embeddings

context augmentation

knowledge retrieval

DO NOT explain in paragraphs.

Only concise phrases.

----------------------------------------

3. summary

Write ONE dense paragraph describing the ideal candidate.

Keep all important technologies.

==========================

Return ONLY JSON.

Do NOT wrap inside markdown.

Do NOT write explanations.

"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt + "\n\nJOB DESCRIPTION:\n\n" + job_description,
        config={
            "response_mime_type": "application/json"
        }
    )

    raw_text = response.text

    cleaned = clean_llm_output(raw_text)

    try:
        return json.loads(cleaned)

    except json.JSONDecodeError:

        print("\n========== RAW GEMINI OUTPUT ==========\n")
        print(raw_text)
        print("\n=======================================\n")

        raise


def processed_jd(jd_file_path):
    """
    Returns

    {
        "career": "...",
        "skills": "...",
        "summary": "..."
    }
    """
    return req_from_jd(jd_file_path)


# -------------------------------
# Debug
# -------------------------------

import sys

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python jd_requirements.py <jd_file_path>")
        sys.exit(1)

    jd_data = processed_jd(sys.argv[1])

    print(json.dumps(jd_data, indent=4))

    print("\nCareer:\n")
    print(jd_data["career"])

    print("\nSkills:\n")
    print(jd_data["skills"])

    print("\nEducation:\n")
    print(jd_data["education"])

    print("\nSummary:\n")
    print(jd_data["summary"])
