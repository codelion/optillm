import re
from typing import Tuple, List
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

SLUG = "readurls"

def extract_urls(text: str) -> List[str]:
    # Updated regex pattern to be more precise
    url_pattern = re.compile(r'https?://[^\s\'"]+')
    
    # Find all matches
    urls = url_pattern.findall(text)
    
    # Clean up the URLs
    cleaned_urls = []
    for url in urls:
        # Remove trailing punctuation and quotes
        url = re.sub(r'[,\'\"\)\]]+$', '', url)
        cleaned_urls.append(url)
    
    return cleaned_urls

def fetch_webpage_content(url: str, max_length: int = 100000) -> str:
    try:
        headers = {
            'User-Agent': 'optillm/0.0.1 (hhttps://github.com/codelion/optillm)'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Make a soup 
        soup = BeautifulSoup(response.content, 'lxml')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text from various elements
        text_elements = []
        
        # Prioritize content from main content tags
        for tag in ['article', 'main', 'div[role="main"]', '.main-content']:
            content = soup.select_one(tag)
            if content:
                text_elements.extend(content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p']))
                break
        
        # If no main content found, fall back to all headers and paragraphs
        if not text_elements:
            text_elements = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p'])
        
        # Extract text from elements
        text = ' '.join(element.get_text(strip=True) for element in text_elements)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove footnote superscripts in brackets
        text = re.sub(r"\[.*?\]+", '', text)
        
        # Truncate to max_length
        if len(text) > max_length:
            text = text[:max_length] + '...'
        
        return text
    except Exception as e:
        return f"Error fetching content: {str(e)}"

def run(system_prompt, initial_query: str, client=None, model=None) -> Tuple[str, int]:
    urls = extract_urls(initial_query)
    # print(urls)
    modified_query = initial_query

    for url in urls:
        content = fetch_webpage_content(url)
        domain = urlparse(url).netloc
        modified_query = modified_query.replace(url, f"{url} [Content from {domain}: {content}]")
    # print(modified_query)
    return modified_query, 0