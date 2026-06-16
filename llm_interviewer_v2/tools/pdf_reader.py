"""
pdf_reader.py — PDF 简历文本提取工具

三级回退策略：
1. pypdf  （纯 Python，速度快）
2. pdfminer.six （布局感知，处理复杂排版）
3. PyMuPDF / fitz （兼容性最强，最后兜底）

用法：
    from tools.pdf_reader import extract_pdf_text
    text = extract_pdf_text(file_bytes)   # bytes → str
"""

from __future__ import annotations
import io


def extract_pdf_text(file_bytes: bytes) -> str:
    """
    从 PDF 字节流中提取纯文本。
    依次尝试 pypdf → pdfminer.six → PyMuPDF，任一成功即返回。
    全部失败则抛出 RuntimeError。
    """
    errors = []

    # ── 第一级：pypdf ─────────────────────────────────────────────
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(pages).strip()
        if text:
            return text
        errors.append("pypdf: 提取内容为空")
    except ImportError:
        errors.append("pypdf: 未安装")
    except Exception as e:
        errors.append(f"pypdf: {e}")

    # ── 第二级：pdfminer.six ──────────────────────────────────────
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract
        text = pdfminer_extract(io.BytesIO(file_bytes)).strip()
        if text:
            return text
        errors.append("pdfminer: 提取内容为空")
    except ImportError:
        errors.append("pdfminer: 未安装")
    except Exception as e:
        errors.append(f"pdfminer: {e}")

    # ── 第三级：PyMuPDF (fitz) ────────────────────────────────────
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages = [page.get_text() for page in doc]
        text = "\n".join(pages).strip()
        if text:
            return text
        errors.append("PyMuPDF: 提取内容为空")
    except ImportError:
        errors.append("PyMuPDF: 未安装")
    except Exception as e:
        errors.append(f"PyMuPDF: {e}")

    raise RuntimeError(
        "所有 PDF 解析方案均失败，请确保安装了 pypdf / pdfminer.six / PyMuPDF 之一。\n"
        + "\n".join(f"  · {e}" for e in errors)
    )