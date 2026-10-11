"""
Markdown to HTML Compiler with MathJax & Modern Styling
File: src/render_docs_to_html.py

Converts all project markdown files into human-readable HTML documents
with rendered LaTeX equations, tables, code blocks, and print-friendly styling.
"""

import os
import re
import markdown

DOCS_LIST = [
    ("docs/MASTER_VIVA_DOCUMENT.md", "MASTER VIVA DOCUMENT - Everything"),
    ("PROJECT_PLAN.md", "Project Plan & Team Roles"),
    ("LEADERS_VIVA_GUIDE.md", "Leader's Viva & Review Guide"),
    ("CODE_EXPLANATION.md", "Code & Parameters Deep Dive"),
    ("docs/LOGISTIC_REGRESSION_PARAMETERS.md", "Logistic Regression Parameters Guide"),
    ("docs/DECISION_TREE_PARAMETERS.md", "Decision Tree Parameters Guide"),
    ("docs/RANDOM_FOREST_PARAMETERS.md", "Random Forest Parameters Guide"),
    ("docs/SVM_PARAMETERS.md", "SVM Parameters Guide"),
    ("docs/PARAMETERS_CHEAT_SHEET.md", "Master Parameters Cheat Sheet")
]

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} — HDFC Bank AI Project</title>
    
    <!-- MathJax for rendering LaTeX formulas -->
    <script>
    MathJax = {{
      tex: {{
        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
        displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
        processEscapes: true
      }},
      svg: {{
        fontCache: 'global'
      }}
    }};
    </script>
    <script type="text/javascript" id="MathJax-script" async
      src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
    </script>
    
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
            line-height: 1.65;
            color: #24292f;
            background-color: #f6f8fa;
            margin: 0;
            padding: 0;
        }}
        .navbar {{
            background-color: #0d47a1;
            color: white;
            padding: 12px 24px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
            position: sticky;
            top: 0;
            z-index: 1000;
        }}
        .navbar h2 {{
            margin: 0;
            font-size: 1.15rem;
            font-weight: 600;
        }}
        .nav-links a {{
            color: #e3f2fd;
            text-decoration: none;
            margin-left: 16px;
            font-size: 0.9rem;
            font-weight: 500;
        }}
        .nav-links a:hover {{
            color: white;
            text-decoration: underline;
        }}
        .container {{
            max-width: 980px;
            margin: 30px auto;
            background: #ffffff;
            padding: 45px 55px;
            border-radius: 8px;
            border: 1px solid #d0d7de;
            box-shadow: 0 3px 12px rgba(140, 149, 159, 0.15);
        }}
        h1, h2, h3, h4 {{
            color: #1a202c;
            font-weight: 600;
            margin-top: 24px;
            margin-bottom: 16px;
            line-height: 1.25;
        }}
        h1 {{
            font-size: 2rem;
            border-bottom: 2px solid #eaecef;
            padding-bottom: 10px;
            color: #0d47a1;
        }}
        h2 {{
            font-size: 1.5rem;
            border-bottom: 1px solid #eaecef;
            padding-bottom: 8px;
            margin-top: 32px;
        }}
        h3 {{
            font-size: 1.25rem;
        }}
        p, ul, ol {{
            margin-top: 0;
            margin-bottom: 16px;
            font-size: 1rem;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            display: table;
            font-size: 0.92rem;
        }}
        th, td {{
            border: 1px solid #d0d7de;
            padding: 10px 14px;
            text-align: left;
        }}
        th {{
            background-color: #f3f4f6;
            font-weight: 600;
        }}
        tr:nth-child(even) {{
            background-color: #f8fafc;
        }}
        code {{
            font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
            background-color: rgba(175, 184, 193, 0.2);
            padding: 0.2em 0.4em;
            border-radius: 6px;
            font-size: 85%;
            color: #b91c1c;
        }}
        pre {{
            background-color: #1e293b;
            color: #f8fafc;
            padding: 16px;
            border-radius: 8px;
            overflow: auto;
            font-size: 88%;
            line-height: 1.45;
        }}
        pre code {{
            background-color: transparent;
            color: inherit;
            padding: 0;
        }}
        blockquote {{
            margin: 16px 0;
            padding: 12px 18px;
            border-left: 4px solid #2563eb;
            background-color: #eff6ff;
            color: #1e3a8a;
            border-radius: 0 6px 6px 0;
        }}
        blockquote p {{
            margin: 0;
        }}
        /* MathJax equation display formatting */
        .MathJax {{
            font-size: 1.15em !important;
            color: #0f172a;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #e2e8f0;
            font-size: 0.85rem;
            color: #64748b;
        }}
        @media print {{
            .navbar {{ display: none; }}
            .container {{ border: none; box-shadow: none; margin: 0; padding: 0; }}
        }}
    </style>
</head>
<body>
    <div class="navbar">
        <h2>🏦 HDFC Bank Loan Prediction — AI Project Manual</h2>
        <div class="nav-links">
            <a href="INDEX.html">📑 All Documents</a>
            <a href="PROJECT_PLAN.html">Project Plan</a>
            <a href="LEADERS_VIVA_GUIDE.html">Viva Guide</a>
            <a href="PARAMETERS_CHEAT_SHEET.html">Cheat Sheet</a>
        </div>
    </div>
    <div class="container">
        {content}
        <div class="footer">
            HDFC Bank Loan Approval Classification Project | Built for AI for Engineers
        </div>
    </div>
</body>
</html>
"""


def compile_all():
    os.makedirs("docs_html", exist_ok=True)
    index_items = []

    md = markdown.Markdown(extensions=['tables', 'fenced_code', 'toc'])

    for path, title in DOCS_LIST:
        if not os.path.exists(path):
            print(f"Skipping missing file: {path}")
            continue

        with open(path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        # Render markdown to HTML
        body_html = md.convert(raw_text)
        md.reset()

        out_name = os.path.basename(path).replace(".md", ".html")
        out_path = os.path.join("docs_html", out_name)

        final_html = HTML_TEMPLATE.format(title=title, content=body_html)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(final_html)

        index_items.append((out_name, title, path))
        print(f"Compiled: {path} -> {out_path}")

    # Build an INDEX.html landing page
    index_content = "<h1>📚 Project Knowledge Base & Documentation</h1>"
    index_content += "<p>Click any guide below to open it in your browser with fully rendered formulas, tables, and study notes:</p>"
    index_content += "<ul>"
    for html_file, title, orig_path in index_items:
        index_content += f"<li style='margin-bottom: 12px; font-size: 1.05rem;'><strong><a href='{html_file}'>{title}</a></strong> <span style='color: #64748b;'>({orig_path})</span></li>"
    index_content += "</ul>"

    index_html = HTML_TEMPLATE.format(title="Project Document Index", content=index_content)
    with open("docs_html/INDEX.html", "w", encoding="utf-8") as f:
        f.write(index_html)
    print("Created: docs_html/INDEX.html")


if __name__ == "__main__":
    compile_all()
