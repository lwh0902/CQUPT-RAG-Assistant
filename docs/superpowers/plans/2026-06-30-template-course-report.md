# Template Course Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the CQUPT-RAG course report with the official template cover, updated chapter requirements, exact typography, verified references, and newly captured real screenshots.

**Architecture:** Treat the legacy `.doc` as a visual source rather than editing it in place, because conversion introduces broken pagination. Extract the official logo and cover hierarchy, generate a clean DOCX deterministically, then render every page for visual QA.

**Tech Stack:** LibreOffice conversion, bundled Python/python-docx/Pillow, in-app browser automation, official web sources, canonical DOCX renderer.

---

### Task 1: Audit and Reuse Official Template Assets

**Files:**
- Read: `/Users/luoweihao/Library/Containers/com.tencent.xinWeChat/Data/Documents/xwechat_files/wxid_yxrenwinptsr22_63ab/temp/RWTemp/2026-06/db3b5c4331c017ed3a70d5e90dd6a4f6/《软件前沿技术》课程期末大作业模版.doc`
- Create: `report_assets/template/official-cover-logo.png`
- Create: `report_assets/template/template-audit.md`

- [ ] Convert the legacy DOC to a temporary DOCX and render all pages.
- [ ] Extract the official CQUPT logo from the converted document package.
- [ ] Record the cover hierarchy, fields, page size, margins, and required typography.
- [ ] Exclude instruction, plagiarism-definition, grading, and sample-case pages from the final submission.

### Task 2: Capture New Real Application Screenshots

**Files:**
- Create: `report_assets/screenshots-v2/01-login.png`
- Create: `report_assets/screenshots-v2/02-chat-home.png`
- Create: `report_assets/screenshots-v2/03-rag-answer.png`
- Create: `report_assets/screenshots-v2/04-knowledge-base.png`
- Create: `report_assets/screenshots-v2/05-pdf-preview.png`

- [ ] Start the current React frontend on a dedicated port and confirm the FastAPI backend.
- [ ] Capture the real login page at a consistent desktop viewport.
- [ ] Authenticate with a local demonstration account and capture the chat home page.
- [ ] Submit a campus-policy question and capture the completed answer with page sources.
- [ ] Open the knowledge-base list and a real PDF page preview, then capture both states.

### Task 3: Verify Evidence and References

**Files:**
- Read: `manual_index_meta.json`
- Read: `experiment_report.json`
- Read: `test_cases.json`
- Read: `report_assets/evidence/references.json`
- Create: `report_assets/evidence/references-v2.json`

- [ ] Confirm the implemented model, embedding, document count, page count, chunk count, and retrieval strategies against code and metadata.
- [ ] Use the 12-case experiment results and explicitly exclude the unexplained old 90% summary.
- [ ] Verify each cited paper or official documentation URL resolves directly and matches its title.
- [ ] Store only links returning a successful browser response or a verified publisher page.

### Task 4: Build the Template-Compliant DOCX

**Files:**
- Create: `report_tools/build_template_course_report.py`
- Create: `report_tools/validate_template_course_report.py`
- Create: `203214445罗炜皓_基于RAG与Function Calling的校园制度智能问答系统设计与实现_模板版.docx`

- [ ] Recreate the official cover with logo, 2025—2026 second semester, 2023 grade, and supplied personal information.
- [ ] Define Heading 1 as 16 pt Heiti, Heading 2 as 14 pt Heiti, Heading 3 as 10.5 pt Songti, and body as 10.5 pt Songti with exactly 20 pt line spacing; set Latin text to Times New Roman.
- [ ] Write original project-specific chapters covering background, objectives, requirements, technology selection, system design, development process, real operation, testing and optimization, limitations, conclusion, and references.
- [ ] Insert five real screenshots, architecture/evaluation figures, captions, and clickable reference hyperlinks.
- [ ] Run structural validation for cover fields, required chapters, font declarations, fixed spacing, image count, hyperlinks, experiment values, and placeholder absence.

### Task 5: Render and Verify Every Page

**Files:**
- Create: `report_assets/rendered-template-final/page-*.png`
- Modify: `report_tools/build_template_course_report.py` if any defect is found
- Modify: final DOCX through deterministic regeneration

- [ ] Render the DOCX using the documents skill's canonical renderer.
- [ ] Inspect every page at original resolution for clipping, overlap, font substitution, table crowding, screenshot readability, and abnormal page breaks.
- [ ] Fix defects, regenerate, rerun structural validation, and rerender until every page is clean.
- [ ] Recheck every reference link and deliver only the final verified DOCX.

