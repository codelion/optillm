import re
import time
import json
import random
from typing import Tuple, List, Dict, Optional
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
from urllib.parse import quote_plus

SLUG = "web_search"

class GoogleSearcher:
    def __init__(self, headless: bool = False, timeout: int = 30):
        self.timeout = timeout
        self.headless = headless
        self.driver = None
        self.setup_driver(headless)
    
    def setup_driver(self, headless: bool = False):
        """Setup Chrome driver with appropriate options"""
        try:
            chrome_options = Options()
            if headless:
                chrome_options.add_argument("--headless")
            else:
                # Non-headless mode - position window for visibility
                chrome_options.add_argument("--window-size=1280,800")
                chrome_options.add_argument("--window-position=100,100")
            
            # Common options
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            
            # More human-like settings
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--disable-web-security")
            chrome_options.add_argument("--disable-features=VizDisplayCompositor")
            chrome_options.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
            
            # Use webdriver-manager to automatically manage ChromeDriver
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            self.driver.set_page_load_timeout(self.timeout)
        except Exception as e:
            raise Exception(f"Failed to setup Chrome driver: {str(e)}")
    
    def detect_captcha(self) -> bool:
        """Detect if CAPTCHA is present on the page"""
        try:
            # Check for common CAPTCHA indicators
            page_source = self.driver.page_source.lower()
            captcha_indicators = [
                'recaptcha',
                'captcha',
                'are you a robot',
                'not a robot',
                'unusual traffic',
                'automated requests',
                'verify you\'re human',
                'verify that you\'re not a robot'
            ]
            
            for indicator in captcha_indicators:
                if indicator in page_source:
                    return True
            
            # Check for reCAPTCHA iframe
            try:
                self.driver.find_element(By.CSS_SELECTOR, "iframe[src*='recaptcha']")
                return True
            except:
                pass
            
            # Check for CAPTCHA challenge div
            try:
                self.driver.find_element(By.ID, "captcha")
                return True
            except:
                pass
            
            return False
        except:
            return False
    
    def wait_for_captcha_resolution(self, max_wait: int = 120) -> bool:
        """Wait for CAPTCHA to be resolved"""
        print("CAPTCHA detected! Please solve it in the browser window.")
        print(f"Waiting up to {max_wait} seconds for CAPTCHA to be solved...")
        
        start_time = time.time()
        check_interval = 2  # Check every 2 seconds
        
        while time.time() - start_time < max_wait:
            time.sleep(check_interval)
            
            # Check if we're still on CAPTCHA page
            if not self.detect_captcha():
                # Check if we have search results
                try:
                    self.driver.find_element(By.CSS_SELECTOR, "div.g")
                    print("CAPTCHA solved! Continuing with search...")
                    return True
                except:
                    # Might be on Google homepage, not CAPTCHA
                    pass
            
            remaining = int(max_wait - (time.time() - start_time))
            if remaining % 10 == 0 and remaining > 0:
                print(f"Still waiting... {remaining} seconds remaining")
        
        print("Timeout waiting for CAPTCHA resolution")
        return False
    
    def search(self, query: str, num_results: int = 10, delay_seconds: Optional[int] = None) -> List[Dict[str, str]]:
        """Perform Google search and return results"""
        if not self.driver:
            raise Exception("Chrome driver not initialized")
        
        try:
            print(f"Searching for: {query}")
            if not self.headless:
                print("Browser window opened")
            
            # First navigate to Google homepage
            self.driver.get("https://www.google.com")
            
            # Wait for page to load and check for CAPTCHA
            time.sleep(1)
            
            # Check if we hit a CAPTCHA immediately
            if self.detect_captcha():
                if not self.wait_for_captcha_resolution():
                    return []
            
            # Check for consent form or cookie banner and accept if present
            try:
                accept_button = self.driver.find_element(By.XPATH, "//button[contains(text(), 'Accept') or contains(text(), 'I agree') or contains(text(), 'Agree')]")
                accept_button.click()
                time.sleep(1)
            except:
                pass  # No consent form
            
            # Find search box and enter query
            try:
                # Try multiple selectors for the search box
                search_box = None
                for selector in [(By.NAME, "q"), (By.CSS_SELECTOR, "input[type='text']"), (By.CSS_SELECTOR, "textarea[name='q']")]:
                    try:
                        search_box = WebDriverWait(self.driver, 5).until(
                            EC.presence_of_element_located(selector)
                        )
                        break
                    except:
                        continue
                
                if search_box:
                    # Use ActionChains for more reliable input
                    actions = ActionChains(self.driver)
                    actions.move_to_element(search_box)
                    actions.click()
                    actions.pause(0.5)
                    # Clear existing text
                    search_box.clear()
                    actions.send_keys(query)
                    actions.pause(0.5)
                    actions.send_keys(Keys.RETURN)
                    actions.perform()
                    
                    # Wait briefly for page to start loading
                    time.sleep(1)
                    
                    # Check for CAPTCHA after search submission
                    if self.detect_captcha():
                        if not self.wait_for_captcha_resolution():
                            return []
                else:
                    raise Exception("Could not find search box")
            except:
                # Fallback to direct URL navigation
                print("Using direct URL navigation...")
                search_url = f"https://www.google.com/search?q={quote_plus(query)}&num={num_results}"
                self.driver.get(search_url)
                time.sleep(1)
                
                # Check for CAPTCHA on direct navigation
                if self.detect_captcha():
                    if not self.wait_for_captcha_resolution():
                        return []
            
            # Wait for search results
            wait = WebDriverWait(self.driver, 10)
            try:
                wait.until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "div.g, [data-sokoban-container], div[data-async-context]"))
                )
            except TimeoutException:
                # Check if it's a CAPTCHA page
                if self.detect_captcha():
                    if self.wait_for_captcha_resolution():
                        # Try waiting for results again after CAPTCHA
                        try:
                            wait.until(
                                EC.presence_of_element_located((By.CSS_SELECTOR, "div.g"))
                            )
                        except:
                            print("No results found after CAPTCHA resolution")
                            return []
                    else:
                        return []
                else:
                    print("Timeout waiting for search results")
                    return []
            
            results = []
            
            # Apply delay AFTER search results are loaded
            if delay_seconds is None:
                delay_seconds = random.randint(8, 64)
            
            if delay_seconds > 0:
                print(f"Applying {delay_seconds} second delay after search...")
                time.sleep(delay_seconds)
            
            print("Extracting search results...")
            
            # Wait for search results to be present
            try:
                print("Waiting for search results to load...")
                # Wait for either the search container or the results themselves
                WebDriverWait(self.driver, 10).until(
                    lambda driver: driver.find_elements(By.CSS_SELECTOR, "div.g") or 
                                   driver.find_element(By.ID, "search") or
                                   driver.find_elements(By.CSS_SELECTOR, "[data-sokoban-container]")
                )
            except TimeoutException:
                print("Timeout waiting for search results. Possible CAPTCHA.")
                if not self.headless:
                    input("Please solve the CAPTCHA if present and press Enter to continue...")
                    # Try waiting again after CAPTCHA
                    try:
                        WebDriverWait(self.driver, 10).until(
                            lambda driver: driver.find_elements(By.CSS_SELECTOR, "div.g")
                        )
                    except:
                        print("Still no results after CAPTCHA attempt")
                        return []
            
            # Debug: Print current URL and page title
            print(f"Current URL: {self.driver.current_url}")
            print(f"Page title: {self.driver.title}")
            
            # Extract search results - try multiple selectors
            search_results = []
            
            # First try the standard div.g selector
            search_results = self.driver.find_elements(By.CSS_SELECTOR, "div.g")
            print(f"Found {len(search_results)} results with div.g")
            
            # If no results, try alternative selectors
            if not search_results:
                # Try finding any element with data-hveid attribute (Google result containers)
                all_elements = self.driver.find_elements(By.CSS_SELECTOR, "[data-hveid]")
                print(f"Found {len(all_elements)} elements with data-hveid")
                
                # Filter to only those that have both h3 and a tags
                for elem in all_elements:
                    try:
                        # Check if this element has both h3 and a link
                        h3 = elem.find_element(By.TAG_NAME, "h3")
                        link = elem.find_element(By.CSS_SELECTOR, "a[href]")
                        if h3 and link:
                            search_results.append(elem)
                    except:
                        continue
                
                print(f"Filtered to {len(search_results)} valid result elements")
            
            if not search_results:
                print("No search results found with any method")
                # Debug: print some page source to see what we're getting
                print("Page source sample (first 500 chars):")
                print(self.driver.page_source[:500])
                return []
            
            # Limit processing to requested number of results
            results_to_process = min(len(search_results), num_results)
            print(f"Processing {results_to_process} results...")
            
            for i, result in enumerate(search_results[:results_to_process]):
                try:
                    # Skip if we already have enough results
                    if len(results) >= num_results:
                        break
                    
                    # Use the same extraction logic as getstars
                    try:
                        url = result.find_element(By.CSS_SELECTOR, "a").get_attribute("href")
                        title = result.find_element(By.CSS_SELECTOR, "h3").text
                        
                        # Skip Google internal URLs
                        if not url or "google.com" in url:
                            continue
                            
                        # Try to get snippet
                        snippet = ""
                        try:
                            # Try multiple snippet selectors
                            snippet_selectors = [".VwiC3b", ".aCOpRe", ".IsZvec"]
                            for selector in snippet_selectors:
                                try:
                                    snippet_elem = result.find_element(By.CSS_SELECTOR, selector)
                                    if snippet_elem and snippet_elem.text:
                                        snippet = snippet_elem.text
                                        break
                                except:
                                    pass
                        except:
                            pass
                        
                        # Add result
                        results.append({
                            "title": title,
                            "url": url,
                            "snippet": snippet or "No description available"
                        })
                        
                        print(f"Extracted result {len(results)}: {title[:50]}...")
                        
                    except NoSuchElementException:
                        print(f"Failed to parse result {i+1}")
                        continue
                        
                except Exception as e:
                    # Skip problematic results
                    continue
            
            # Deduplicate results by URL
            seen_urls = set()
            unique_results = []
            for result in results:
                if result["url"] not in seen_urls:
                    seen_urls.add(result["url"])
                    unique_results.append(result)
            
            print(f"Successfully extracted {len(unique_results)} unique search results (from {len(results)} total)")
            
            return unique_results
            
        except TimeoutException as e:
            # Return empty results instead of raising
            print(f"Search timeout for query '{query}': {str(e)}")
            return []
        except WebDriverException as e:
            print(f"WebDriver error during search: {str(e)}")
            return []
        except Exception as e:
            print(f"Unexpected error during search: {str(e)}")
            return []
    
    def close(self):
        """Close the browser driver"""
        if self.driver:
            self.driver.quit()
            self.driver = None

def extract_search_queries(text: str) -> List[str]:
    """Extract potential search queries from the input text"""
    # Clean up common prefixes from chat messages
    text = text.strip()
    # Remove common role prefixes
    for prefix in ["User:", "user:", "User ", "user ", "Assistant:", "assistant:", "System:", "system:"]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    
    # Look for explicit search requests
    search_patterns = [
        r"search for[:\s]+([^\n\.]+)",
        r"find information about[:\s]+([^\n\.]+)",
        r"look up[:\s]+([^\n\.]+)",
        r"research[:\s]+([^\n\.]+)",
    ]
    
    queries = []
    for pattern in search_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        queries.extend([match.strip() for match in matches])
    
    # If no explicit patterns, use the text as a search query
    if not queries:
        # Remove question marks and clean up
        cleaned_query = text.replace("?", "").strip()
        # If it looks like a question or search query, use it
        if cleaned_query and len(cleaned_query.split()) > 2:
            queries.append(cleaned_query)
        else:
            # Clean up the text to make it search-friendly
            cleaned_query = re.sub(r'[^\w\s]', ' ', text)
            cleaned_query = ' '.join(cleaned_query.split())
            if len(cleaned_query) > 100:
                # Take first 100 characters
                cleaned_query = cleaned_query[:100].rsplit(' ', 1)[0]
            if cleaned_query:
                queries.append(cleaned_query)
    
    return queries

def format_search_results(query: str, results: List[Dict[str, str]]) -> str:
    """Format search results into readable text"""
    if not results:
        return f"No search results found for: {query}"
    
    formatted = f"Search results for '{query}':\n\n"
    
    for i, result in enumerate(results, 1):
        formatted += f"{i}. **{result['title']}**\n"
        formatted += f"   URL: {result['url']}\n"
        if result['snippet']:
            formatted += f"   Summary: {result['snippet']}\n"
        formatted += "\n"
    
    return formatted

def run(system_prompt: str, initial_query: str, client=None, model: str = None, request_config: Optional[Dict] = None) -> Tuple[str, int]:
    """
    Web search plugin that uses Chrome to search Google and return results
    
    Args:
        system_prompt: System prompt for the conversation
        initial_query: User's query that may contain search requests
        client: OpenAI client (unused for this plugin)
        model: Model name (unused for this plugin) 
        request_config: Optional configuration dict with keys:
            - num_results: Number of search results (default: 10)
            - delay_seconds: Delay between searches in seconds (default: random 8-64)
                            Set to 0 to disable delays, or specify exact seconds
            - headless: Run browser in headless mode (default: False)
            - timeout: Browser timeout in seconds (default: 30)
    
    Returns:
        Tuple of (enhanced_query_with_search_results, completion_tokens)
    """
    # Parse configuration
    config = request_config or {}
    num_results = config.get("num_results", 10)
    delay_seconds = config.get("delay_seconds", None)  # None means random 32-128
    headless = config.get("headless", False)  # Default to non-headless
    timeout = config.get("timeout", 30)  # Standard timeout
    
    # Extract search queries from the input
    search_queries = extract_search_queries(initial_query)
    
    if not search_queries:
        return initial_query, 0
    
    searcher = None
    try:
        searcher = GoogleSearcher(headless=headless, timeout=timeout)
        enhanced_query = initial_query
        
        for query in search_queries:
            # Perform the search
            results = searcher.search(query, num_results=num_results, delay_seconds=delay_seconds)
            
            # Format results
            if results:
                formatted_results = format_search_results(query, results)
                # Append results to the query
                enhanced_query = f"{enhanced_query}\n\n[Web Search Results]:\n{formatted_results}"
            else:
                # No results found - add a note
                enhanced_query = f"{enhanced_query}\n\n[Web Search Results]:\nNo results found for '{query}'. This may be due to network issues or search restrictions."
        
        return enhanced_query, 0
        
    except Exception as e:
        error_msg = f"Web search error: {str(e)}"
        enhanced_query = f"{initial_query}\n\n[Web Search Error]: {error_msg}"
        return enhanced_query, 0
        
    finally:
        if searcher:
            searcher.close()