from pypdf import PdfReader
from docx import Document
import io
import re


def extract_text_from_pdf(file_content: bytes) -> str:
    try:
        pdf_file = io.BytesIO(file_content)
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        raise ValueError(f"Error processing PDF: {str(e)}")


def extract_text_from_docx(file_content: bytes) -> str:
    try:
        docx_file = io.BytesIO(file_content)
        doc = Document(docx_file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    except Exception as e:
        raise ValueError(f"Error processing DOCX: {str(e)}")


def extract_text_from_txt(file_content: bytes) -> str:
    try:
        return file_content.decode('utf-8')
    except Exception as e:
        raise ValueError(f"Error processing TXT: {str(e)}")


def process_file(filename: str, file_content: bytes) -> str:
    filename_lower = filename.lower()
    if filename_lower.endswith('.pdf'):
        return extract_text_from_pdf(file_content)
    elif filename_lower.endswith('.docx'):
        return extract_text_from_docx(file_content)
    elif filename_lower.endswith('.txt'):
        return extract_text_from_txt(file_content)
    else:
        raise ValueError(f"Unsupported file type. Supported: PDF, DOCX, TXT")


# Curated header keywords for resumes + generic docs. Expand as your corpus grows.
_HEADER_KEYWORDS = [
    "education", "experience", "work experience", "projects", "skills",
    "problem solving", "certifications", "achievements", "summary",
    "objective", "contact", "languages", "publications", "policy",
    "leadership", "extracurricular", "awards",
]

_SECTION_NUM_RE = re.compile(r'^section\s*\d+', re.IGNORECASE)


def _is_header(line: str) -> bool:
    """
    A line is a header if it's short, doesn't end like a sentence, and is
    DOMINATED by a header keyword — not merely contains one as a substring.
    e.g. "Skills" or "Technical Skills" is a header; "Redis skills, Docker"
    is not, even though both contain "skills". Without this word-count
    check, ordinary bullet lines get misclassified as new section headers
    and silently truncate the real section.
    """
    if not line or len(line) >= 60 or line.endswith(('.', ',', ';')):
        return False

    line_lower = line.lower()

    if _SECTION_NUM_RE.match(line_lower):
        return True

    for kw in _HEADER_KEYWORDS:
        if kw in line_lower:
            kw_words = len(kw.split())
            line_words = len(line_lower.split())
            # keyword must make up most of the line's words
            if line_words <= kw_words + 2:
                return True
    return False


def _split_with_overlap(content: str, chunk_size: int, overlap: int) -> list[str]:
    """Same sliding-window logic as the original chunk_text, scoped to one section."""
    pieces = []
    start = 0
    while start < len(content):
        end = start + chunk_size
        piece = content[start:end]
        if end < len(content):
            last_period = piece.rfind('.')
            last_newline = piece.rfind('\n')
            break_point = max(last_period, last_newline)
            if break_point > chunk_size // 2:
                piece = piece[:break_point + 1]
                end = start + break_point + 1
        piece = piece.strip()
        if piece:
            pieces.append(piece)
        start = end - overlap
    return pieces


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[dict]:
    """
    Section-aware parent-child chunking.

    Parent = a detected section (EDUCATION, PROJECTS, "Section 1: ...", etc.)
    Child  = the section's full text if it fits in chunk_size, otherwise the
             section split into overlapping sub-chunks that never cross a
             section boundary.

    Every child has its parent header prepended (e.g. "[SKILLS]\\n...") so it
    embeds as a dense, standalone semantic unit — this matters for a small
    model like bge-small-en-v1.5, which has no way to infer context for a
    bare list like "FastAPI, Docker, Redis" without it.

    Docs with no detectable headers fall through as a single "GENERAL"
    section, which still gets sub-chunked normally if it's long — so nothing
    breaks for unstructured documents.

    Returns a list of dicts: {"text": ..., "section": ..., "chunk_no": ...}
    """
    lines = text.split('\n')
    sections = []
    current_header = "GENERAL"
    current_content = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if _is_header(line):
            if current_content:
                sections.append({"header": current_header, "text": "\n".join(current_content)})
            current_header = line.upper()
            current_content = []
        else:
            current_content.append(line)

    if current_content:
        sections.append({"header": current_header, "text": "\n".join(current_content)})

    final_chunks = []
    chunk_idx = 0
    for sec in sections:
        header = sec["header"]
        content = sec["text"]

        if len(content) <= chunk_size:
            final_chunks.append({
                "text": f"[{header}]\n{content}",
                "section": header,
                "chunk_no": chunk_idx,
            })
            chunk_idx += 1
        else:
            for piece in _split_with_overlap(content, chunk_size, overlap):
                final_chunks.append({
                    "text": f"[{header}]\n{piece}",
                    "section": header,
                    "chunk_no": chunk_idx,
                })
                chunk_idx += 1

    return final_chunks