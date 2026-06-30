import os
import sys
import time
import json
import base64
import requests
from bs4 import BeautifulSoup
from anthropic import Anthropic
import db

PDF_FILE = "portfolio.pdf"
MODEL_NAME = "claude-3-5-sonnet-20240620"

# Will be initialized inside process_jobs
client = None

SYSTEM_PROMPT = """
You are a job search assistant helping an F1 international student named Atharva Patil find entry-level jobs in the USA.

Candidate profile summary:
- MS Data Science @ CU Boulder (graduating May 2026), GPA 3.9
- Skills: Python, R, SQL, ML/DL, NLP, Computer Vision, ETL, FastAPI, AWS, Power BI, Tableau
- Target roles: Data Science, ML Engineer, Data Engineer, AI Engineer, Analytics, Software Engineer (backend/data), Business Analyst
- NEEDS visa sponsorship (F1 OPT → H1B). Prioritize companies known to sponsor.

You will receive scraped text from a company's career page.

Find up to 3 best-matching job postings. Apply these strict filters:
✓ USA only (remote-USA is fine, international = skip)
✓ Entry-level / new grad only (skip: senior, staff, lead, manager, director, principal)
✓ Must match Atharva's target roles above
✓ Posted within last 60 days if date is visible (skip older)

Return ONLY a raw JSON array, no markdown, no explanation:
[
  {
    "job_title": "...",
    "apply_link": "...",
    "location": "...",
    "sponsorship": "Yes | No | Not Mentioned",
    "entry_level": "Yes | No",
    "date_posted": "YYYY-MM-DD or Not Listed",
    "match_score": 8,
    "notes": "one sentence on why this fits Atharva"
  }
]

If zero jobs match all filters, return exactly: []
"""

def scrape_url(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        for el in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            el.decompose()

        text_parts = []
        for element in soup.descendants:
            if isinstance(element, str):
                cleaned = element.strip()
                if cleaned:
                    text_parts.append(cleaned)
            elif element.name == "a" and element.get("href"):
                text_parts.append(f"({element.get('href')})")
                
        text = " ".join(text_parts)
        return text[:6000]
    except Exception as e:
        return None

def analyze_with_claude(scraped_text, pdf_text):
    global client
    
    models = [
        "claude-sonnet-4-6",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-sonnet-20240620",
        "claude-3-5-sonnet-latest",
        "claude-3-sonnet-20240229",
        "claude-3-opus-20240229",
        "claude-3-haiku-20240307",
        "claude-3-5-haiku-20241022",
        "claude-3-5-haiku-latest",
        "claude-3-7-sonnet-20250219",
        "claude-4-sonnet-20250514",
        "claude-4-sonnet-latest",
        "claude-4-6-sonnet-20260228",
        "claude-2.1",
        "claude-2.0"
    ]
    
    try:
        if not client:
            client = Anthropic()
        
        last_error = None
        for model in models:
            try:
                message = client.messages.create(
                    model=model,
                    max_tokens=1000,
                    system=SYSTEM_PROMPT,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"Candidate Resume:\n\n{pdf_text}\n\nScraped career page text for analysis:\n\n{scraped_text}"
                                }
                            ]
                        }
                    ]
                )
                content = message.content[0].text.strip()
                start_idx = content.find('[')
                end_idx = content.rfind(']')
                if start_idx != -1 and end_idx != -1:
                    content = content[start_idx:end_idx+1]
                return json.loads(content)
            except Exception as e:
                error_str = str(e)
                last_error = e
                # Only retry if it's a 404 model not found
                if "404" in error_str and "model" in error_str:
                    print(f"Model {model} not found, trying next...")
                    continue
                else:
                    return e
                    
        return last_error
    except Exception as e:
        print(f"Claude API Error: {e}")
        return e

def process_jobs():
    """Scan ALL pending companies."""
    global client
    
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return {"status": "error", "message": "ANTHROPIC_API_KEY not set."}
    
    if not os.path.exists(PDF_FILE):
        return {"status": "error", "message": f"{PDF_FILE} not found."}
    
    import PyPDF2
    pdf_text = ""
    with open(PDF_FILE, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pdf_text += text + "\n"
    
    rows = db.get_company_rows_for_scan()
    
    if not rows:
        return {"status": "success", "message": "No new companies to process."}
    
    for row in rows:
        row_id = row["id"]
        company = row["company_name"]
        url = row["career_url"]
        
        print(f"Scanning {company}...")
        
        # 1. Scrape
        text = scrape_url(url)
        if not text or len(text) < 200:
            for suffix in ["/jobs", "/careers", "/openings"]:
                base = url.rstrip('/')
                fallback_text = scrape_url(base + suffix)
                if fallback_text and len(fallback_text) > 200:
                    text = fallback_text
                    break
        
        if not text:
            db.set_job_status(row_id, "Failed to scrape")
            print(f"✗ {company} → Failed to scrape")
            continue
        
        # 2. Claude API
        results = analyze_with_claude(text, pdf_text)
        
        if isinstance(results, Exception):
            db.set_job_status(row_id, f"Error: {results}")
            continue
        
        # 3. Filter out already-applied jobs
        applied_titles = db.get_applied_titles_for_company(company)
        results = [j for j in results if j.get("job_title", "").lower().strip() not in applied_titles]
        
        # 4. Write results
        if len(results) == 0:
            db.set_job_status(row_id, "No matches found")
            print(f"✗ {company} → No matches")
        else:
            print(f"✓ {company} → {len(results)} jobs found")
            # Write first result to existing row
            db.write_job_result(row_id, results[0])
            # Insert extra rows for additional results
            for job in results[1:]:
                db.insert_extra_job(company, url, job)
        
        time.sleep(2)  # polite delay

    return {"status": "success", "message": "Scraping complete!"}

def process_single_company(company_name):
    """Scan a single company by name."""
    global client
    
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return {"status": "error", "message": "ANTHROPIC_API_KEY not set."}
    
    if not os.path.exists(PDF_FILE):
        return {"status": "error", "message": f"{PDF_FILE} not found."}
    
    import PyPDF2
    pdf_text = ""
    with open(PDF_FILE, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pdf_text += text + "\n"
    
    row_info = db.get_company_row_for_single_scan(company_name)
    
    if not row_info:
        return {"status": "error", "message": f"Company '{company_name}' not found in tracker."}
    
    row_id = row_info["id"]
    url = row_info["url"]
    
    # Clear previous results
    db.clear_job_row(row_id)
    
    # Scrape
    text = scrape_url(url)
    if not text or len(text) < 200:
        for suffix in ["/jobs", "/careers", "/openings"]:
            base = url.rstrip('/')
            fallback_text = scrape_url(base + suffix)
            if fallback_text and len(fallback_text) > 200:
                text = fallback_text
                break
    
    if not text:
        db.set_job_status(row_id, "Failed to scrape")
        return {"status": "error", "message": f"Failed to scrape {company_name}"}
    
    results = analyze_with_claude(text, pdf_text)
    
    if isinstance(results, Exception):
        db.set_job_status(row_id, f"Error: {results}")
        return {"status": "error", "message": f"Claude failed: {results}"}
    
    if len(results) == 0:
        db.set_job_status(row_id, "No matches found")
        return {"status": "success", "message": f"{company_name}: No matching jobs found."}
    
    # Filter out already-applied jobs
    applied_titles = db.get_applied_titles_for_company(company_name)
    results = [j for j in results if j.get("job_title", "").lower().strip() not in applied_titles]
    
    if len(results) == 0:
        db.delete_empty_row(row_id)
        return {"status": "success", "message": f"{company_name}: All found jobs were already applied to."}
    
    # Write results
    db.write_job_result(row_id, results[0])
    for job in results[1:]:
        db.insert_extra_job(company_name, url, job)
    
    return {"status": "success", "message": f"{company_name}: {len(results)} job(s) found!"}

if __name__ == "__main__":
    process_jobs()
