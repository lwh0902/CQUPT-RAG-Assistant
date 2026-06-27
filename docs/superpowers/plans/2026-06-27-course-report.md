# Course Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a polished, evidence-based Word course report for the CQUPT-RAG project with verified references and real application screenshots.

**Architecture:** Keep evidence collection, screenshot capture, document generation, and validation as separate stages. Generate the report deterministically from a focused builder script, then render the DOCX to page images and inspect every page before delivery.

**Tech Stack:** Python/OOXML document tooling from the bundled workspace runtime, LibreOffice renderer, in-app browser automation, FastAPI/React project evidence, authoritative web sources.

---

### Task 1: Audit Project Evidence and Reference Sources

**Files:**
- Read: `README.md`
- Read: `experiment_report.json`
- Read: `test_cases.json`
- Read: `rag.py`
- Read: `services/hybrid.py`
- Read: `routers/chat.py`
- Create: `report_assets/evidence/report_facts.md`
- Create: `report_assets/evidence/references.json`

- [ ] **Step 1: Extract implemented features and architecture from code**

Record only claims supported by the listed source files: four retrieval strategies, query rewriting, vector/BM25 retrieval, RRF fusion, Function Calling, WebSocket streaming, memory, authentication, source citations, document listing, and PDF preview.

- [ ] **Step 2: Reconcile evaluation metrics**

Parse `experiment_report.json` and `test_cases.json`, identify the exact number of cases and metric definition, and document why the README and `test_report.md` numbers differ. Use one clearly labeled experiment in the report rather than combining incompatible results.

- [ ] **Step 3: Find authoritative references**

Use primary sources only: original RAG paper, original RRF paper, Sentence-BERT or the embedding model's authoritative publication, official FastAPI/LangChain/Chroma/Zhipu documentation where directly relevant. Store title, authors or organization, year, and direct URL.

- [ ] **Step 4: Verify every reference URL**

Open each URL and confirm that it resolves directly to the cited paper or official documentation page. Exclude search-result pages, blogs, scraped copies, and inaccessible links.

### Task 2: Capture Real Functional Screenshots

**Files:**
- Create: `report_assets/screenshots/01-login-page.png`
- Create: `report_assets/screenshots/02-chat-home.png`
- Create: `report_assets/screenshots/03-knowledge-base.png`
- Create: `report_assets/screenshots/04-pdf-preview.png`
- Create: `report_assets/screenshots/05-rag-answer-with-sources.png`

- [ ] **Step 1: Start the actual frontend and backend**

Verify the React application and FastAPI API are serving the same project and use a dedicated frontend port to avoid capturing another local application.

- [ ] **Step 2: Capture login and authenticated home views**

Use the actual rendered browser pages at a consistent desktop viewport; do not use mockups or source assets.

- [ ] **Step 3: Capture knowledge-base and PDF-preview views**

Open the real knowledge-base modal and the first page of a mounted PDF so document names, page count, and preview content are visible.

- [ ] **Step 4: Capture one completed RAG answer**

Ask a campus-policy question, wait for the streamed answer to finish, verify source citations appear, and capture the complete result without exposing credentials.

### Task 3: Generate the Word Report

**Files:**
- Create: `report_tools/build_course_report.py`
- Create: `report_tools/validate_course_report.py`
- Create: `203214445罗炜皓_基于RAG与Function Calling的校园制度智能问答系统设计与实现.docx`

- [ ] **Step 1: Implement deterministic styles and page geometry**

Use A4 portrait pages, Chinese Songti typography, Times New Roman for Latin text and digits, 1.5-line body spacing, two-character first-line indentation, numbered headings, centered figures, captions, and continuous page numbers.

- [ ] **Step 2: Write evidence-based report content**

Include cover, abstract, keywords, background, frontier technologies, requirements, architecture, core implementation, functional demonstration, experiment analysis, innovations, limitations, conclusion, and references. Describe the project as an application of pretrained models and RAG, not as a newly trained foundation model.

- [ ] **Step 3: Insert and caption verified screenshots**

Insert all five screenshots with readable sizing and captions, keeping each figure with its explanatory paragraph and avoiding page-edge clipping.

- [ ] **Step 4: Add verified hyperlinks to references**

Create clickable hyperlinks from each reference entry to the verified direct source URL.

- [ ] **Step 5: Run structural validation**

Execute `validate_course_report.py` and require checks for personal metadata, all required section titles, all five embedded images, font declarations, reference hyperlinks, and absence of placeholder text.

### Task 4: Render and Visually Verify the Report

**Files:**
- Create: `report_assets/rendered/page-*.png`
- Modify: `report_tools/build_course_report.py` if defects are found
- Modify: final DOCX if regenerated

- [ ] **Step 1: Render the DOCX to PNG pages**

Use the documents skill's canonical `render_docx.py` with a writable output directory and optional PDF emission for inspection.

- [ ] **Step 2: Inspect every rendered page**

Check cover alignment, heading hierarchy, Chinese glyphs, Latin fonts, paragraphs, figure readability, captions, page breaks, references, headers, footers, and page numbers.

- [ ] **Step 3: Fix and rerender until clean**

Adjust image dimensions, keep-with-next settings, spacing, and page breaks where needed; rerun structural validation and render again after every correction.

- [ ] **Step 4: Deliver only the final DOCX**

Confirm the final filename matches the course naming rule and provide the verified document to the user.

