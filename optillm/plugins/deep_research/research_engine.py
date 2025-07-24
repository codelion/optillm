"""
Deep Research Engine - Core Implementation

This module implements the Test-Time Diffusion Deep Researcher (TTD-DR) algorithm
as described in "Deep Researcher with Test-Time Diffusion" (https://arxiv.org/abs/2507.16075v1).

The TTD-DR approach treats research as a diffusion process with iterative refinement
through denoising and retrieval, generating comprehensive research reports.
"""

import asyncio
import json
import re
from typing import Tuple, List, Dict, Optional, Any
from datetime import datetime
from collections import defaultdict
from optillm.plugins.web_search_plugin import run as web_search_run
from optillm.plugins.readurls_plugin import run as readurls_run
from optillm.plugins.memory_plugin import run as memory_run


def clean_reasoning_tags(text: str) -> str:
    """
    Remove reasoning tags from model responses for clean final output.
    
    Removes common reasoning tags like:
    - <think></think>
    - <thinking></thinking>
    - <reasoning></reasoning>
    - <thought></thought>
    
    Args:
        text: Raw model response text
        
    Returns:
        Cleaned text with reasoning tags removed
    """
    if not text:
        return text
    
    # List of reasoning tag patterns to remove
    reasoning_patterns = [
        r'<think>.*?</think>',
        r'<thinking>.*?</thinking>',
        r'<reasoning>.*?</reasoning>',
        r'<thought>.*?</thought>',
        r'<reflect>.*?</reflect>',
        r'<reflection>.*?</reflection>',
    ]
    
    cleaned_text = text
    for pattern in reasoning_patterns:
        # Use DOTALL flag to match across newlines
        cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.DOTALL | re.IGNORECASE)
    
    # Clean up any extra whitespace left behind, but preserve markdown formatting
    cleaned_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_text)  # Multiple empty lines to double
    cleaned_text = re.sub(r'  +', ' ', cleaned_text)  # Multiple spaces to single space (but preserve intentional double spaces)
    cleaned_text = cleaned_text.strip()
    
    return cleaned_text


class DeepResearcher:
    """
    Implementation of Test-Time Diffusion Deep Researcher (TTD-DR) algorithm
    
    This class implements the paper's approach of treating research as a diffusion process
    with iterative refinement through denoising and retrieval.
    
    Based on: https://arxiv.org/abs/2507.16075v1
    """
    
    def __init__(self, client, model: str, max_iterations: int = 5, max_sources: int = 10):
        self.client = client
        self.model = model
        self.max_iterations = max_iterations
        self.max_sources = max_sources
        self.research_state = {
            "queries": [],
            "sources": [],
            "content": [],
            "synthesis": "",
            "iteration": 0
        }
        self.total_tokens = 0
        self.citations = {}  # Map citation number to source info
        self.citation_counter = 0
        self.source_content_map = {}  # Map URL to content for citations
    
    def decompose_query(self, system_prompt: str, initial_query: str) -> List[str]:
        """
        Decompose complex research query into focused sub-queries
        This implements the query planning phase of TTD-DR
        """
        decomposition_prompt = f"""
        You are a research assistant. Given a complex query, break it down into 3-5 focused sub-queries that would help gather comprehensive information.
        
        Original query: {initial_query}
        
        Provide sub-queries in this format:
        1. [specific focused question]
        2. [specific focused question]
        3. [specific focused question]
        ...
        
        Make each sub-query specific and searchable. Focus on different aspects of the main topic.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": decomposition_prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            content = response.choices[0].message.content.strip()
            # Clean reasoning tags from query decomposition response
            content = clean_reasoning_tags(content)
            self.total_tokens += response.usage.completion_tokens
            
            # Extract numbered queries
            queries = []
            for line in content.split('\n'):
                line = line.strip()
                if re.match(r'^\d+\.', line):
                    query = re.sub(r'^\d+\.\s*', '', line).strip()
                    if query:
                        queries.append(query)
            
            return queries[:5]  # Limit to 5 sub-queries
            
        except Exception as e:
            # Fallback: use original query
            return [initial_query]
    
    def perform_web_search(self, queries: List[str]) -> str:
        """
        Perform web search for multiple queries using the web_search plugin
        """
        all_results = []
        
        # Perform individual searches for each query to avoid truncation issues
        for i, query in enumerate(queries):
            try:
                # Format as a clean search query
                search_query = f"search for {query.strip()}"
                
                # Perform search with reduced results per query to stay within limits
                results_per_query = max(1, self.max_sources // len(queries))
                
                enhanced_query, _ = web_search_run("", search_query, None, None, {
                    "num_results": results_per_query,
                    "delay_seconds": 2 if i == 0 else 1,  # Shorter delay for subsequent queries
                    "headless": False  # Allow CAPTCHA solving if needed
                })
                
                if enhanced_query and "Web Search Results" in enhanced_query:
                    all_results.append(enhanced_query)
                    
            except Exception as e:
                # Continue with other queries if one fails
                all_results.append(f"Search failed for query '{query}': {str(e)}")
                continue
        
        if not all_results:
            return "Web search failed: No results obtained from any query"
        
        # Combine all search results
        combined_results = "\n\n".join(all_results)
        return combined_results
    
    def extract_and_fetch_urls(self, search_results: str) -> Tuple[str, List[Dict]]:
        """
        Extract URLs from search results and fetch their content using readurls plugin
        Returns content and list of sources with metadata
        """
        try:
            # First extract URLs and metadata from search results
            sources = []
            
            # Pattern to match search result blocks
            result_pattern = r'(\d+)\.\s*\*\*(.+?)\*\*\s*\n\s*URL:\s*(.+?)\n'
            matches = re.findall(result_pattern, search_results, re.MULTILINE)
            
            for match in matches:
                source = {
                    'number': match[0],
                    'title': match[1].strip(),
                    'url': match[2].strip(),
                    'access_date': datetime.now().strftime('%Y-%m-%d')
                }
                sources.append(source)
            
            # If regex doesn't work, try line-by-line parsing
            if not sources:
                lines = search_results.split('\n')
                current_source = {}
                
                for i, line in enumerate(lines):
                    # Check for numbered item with title
                    title_match = re.match(r'^(\d+)\.\s*\*\*(.+?)\*\*', line.strip())
                    if title_match:
                        if current_source and 'url' in current_source:
                            sources.append(current_source)
                        current_source = {
                            'number': title_match.group(1),
                            'title': title_match.group(2).strip()
                        }
                    # Check for URL line
                    elif line.strip().startswith('URL:') and current_source:
                        url = line.strip()[4:].strip()
                        current_source['url'] = url
                        current_source['access_date'] = datetime.now().strftime('%Y-%m-%d')
                
                if current_source and 'url' in current_source:
                    sources.append(current_source)
            
            # Fetch content for all URLs
            content_with_urls, _ = readurls_run("", search_results, None, None)
            
            return content_with_urls, sources
        except Exception as e:
            return f"URL fetching failed: {str(e)}", []
    
    def synthesize_with_memory(self, system_prompt: str, query: str, content: str, sources: List[Dict]) -> Tuple[str, int]:
        """
        Use memory plugin to synthesize information from collected content with citations
        """
        # Add citation instructions to the synthesis request
        citation_prompt = f"""
        IMPORTANT CITATION INSTRUCTIONS:
        1. Use numbered citations [1], [2], etc. to reference specific sources
        2. Place citations immediately after the relevant fact or quote
        3. Multiple citations can be used together like [1,3,5]
        4. Every major claim, statistic, or specific finding MUST have a citation
        5. When quoting directly, use quotation marks and cite immediately after
        
        Available sources for citation:
        """
        
        # Register sources for citations, avoiding duplicates
        url_to_citation = {}  # Track which URLs already have citations
        
        for source in sources:
            if 'url' in source:
                url = source['url']
                # Check if this URL already has a citation
                if url not in url_to_citation:
                    self.citation_counter += 1
                    self.citations[self.citation_counter] = source
                    url_to_citation[url] = self.citation_counter
                    citation_prompt += f"\n[{self.citation_counter}] {source.get('title', 'Untitled')} - {url}"
        
        # Format content for memory plugin with citation instructions
        memory_input = f"{citation_prompt}\n\n{content}\n\nQuery: {query}\n\nRemember to cite all sources using [1], [2], etc. format throughout your synthesis."
        
        try:
            synthesis, tokens = memory_run(system_prompt, memory_input, self.client, self.model)
            # Clean reasoning tags from synthesis response
            synthesis = clean_reasoning_tags(synthesis)
            return synthesis, tokens
        except Exception as e:
            return f"Memory synthesis failed: {str(e)}", 0
    
    def evaluate_completeness(self, system_prompt: str, query: str, current_synthesis: str) -> Tuple[bool, List[str]]:
        """
        Evaluate if the current research is complete or needs more information
        Returns (is_complete, list_of_missing_aspects)
        """
        evaluation_prompt = f"""
        You are evaluating the completeness of a research synthesis. 
        
        Original query: {query}
        Current synthesis: {current_synthesis}
        
        Evaluate if this synthesis adequately addresses the original query. Consider:
        1. Are all major aspects of the query covered?
        2. Is there sufficient depth and detail?
        3. Are there any obvious gaps or missing information?
        
        Respond in this format:
        COMPLETE: [YES/NO]
        MISSING: [list any missing aspects, one per line, or "None" if complete]
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": evaluation_prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            content = response.choices[0].message.content.strip()
            # Clean reasoning tags from completeness evaluation response
            content = clean_reasoning_tags(content)
            self.total_tokens += response.usage.completion_tokens
            
            # Parse response
            is_complete = "COMPLETE: YES" in content.upper()
            
            missing_aspects = []
            if "MISSING:" in content.upper():
                missing_section = content.split("MISSING:")[-1].strip()
                if missing_section.upper() != "NONE":
                    missing_aspects = [line.strip() for line in missing_section.split('\n') if line.strip()]
            
            return is_complete, missing_aspects
            
        except Exception as e:
            # Default to not complete on error
            return False, ["Error in evaluation"]
    
    def generate_focused_queries(self, missing_aspects: List[str], original_query: str) -> List[str]:
        """
        Generate focused search queries to address missing aspects
        """
        focused_queries = []
        for aspect in missing_aspects:
            # Create a focused query combining the original topic with the missing aspect
            focused_query = f"{original_query} {aspect}"
            focused_queries.append(focused_query)
        
        return focused_queries[:3]  # Limit to 3 additional queries per iteration
    
    def generate_structured_report(self, system_prompt: str, original_query: str, synthesis: str) -> str:
        """
        Generate a properly structured research report with sections and citations
        """
        # Build citation context
        citation_context = "\nAvailable citations:\n"
        for num, source in self.citations.items():
            citation_context += f"[{num}] {source.get('title', 'Untitled')}\n"
        
        report_prompt = f"""
        Generate a comprehensive research report with the following structure:
        
        # Research Report: [Create an appropriate title based on the query]
        
        ## Executive Summary
        [Provide a 2-3 paragraph summary of the key findings and conclusions]
        
        ## 1. Introduction
        [Introduce the research question and its significance]
        
        ## 2. Background
        [Provide necessary context and background information]
        
        ## 3. Key Findings
        [Present the main findings organized by themes or categories]
        
        ## 4. Analysis and Discussion
        [Analyze the findings and their implications]
        
        ## 5. Conclusion
        [Summarize the research and provide final thoughts]
        
        ## 6. Recommendations (if applicable)
        [Provide actionable recommendations based on findings]
        
        ## 7. Limitations and Future Research
        [Acknowledge any limitations and suggest areas for future investigation]
        
        Original query: {original_query}
        
        Research synthesis with citations: {synthesis}
        
        {citation_context}
        
        IMPORTANT INSTRUCTIONS:
        1. Use numbered citations [1], [2], etc. throughout the report to reference sources
        2. Ensure EVERY major claim, statistic, or finding has a citation
        3. Use markdown formatting for structure (## for main sections, ### for subsections)
        4. Be comprehensive but concise (aim for 1500-2500 words)
        5. Maintain academic tone and objectivity
        6. Include specific data, statistics, and examples where available
        7. Use direct quotes sparingly and always with citations
        8. Group related citations together when appropriate [1,2,3]
        9. Ensure the Executive Summary captures the essence of the entire report
        10. Make recommendations specific and actionable
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": report_prompt}
                ],
                temperature=0.7,
                max_tokens=3000  # Increased for comprehensive report
            )
            
            report_content = response.choices[0].message.content.strip()
            # Clean reasoning tags from final report response
            report_content = clean_reasoning_tags(report_content)
            self.total_tokens += response.usage.completion_tokens
            
            # Add references section with proper formatting
            references = "\n\n## References\n\n"
            for num, source in sorted(self.citations.items()):
                title = source.get('title', 'Untitled')
                url = source['url']
                access_date = source.get('access_date', datetime.now().strftime('%Y-%m-%d'))
                
                # Format reference in academic style
                references += f"[{num}] {title}. "
                references += f"Available at: <{url}> "
                references += f"[Accessed: {access_date}]\n\n"
            
            # Add metadata footer
            metadata = "\n---\n\n**Research Metadata:**\n"
            metadata += f"- Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            metadata += f"- Research iterations: {self.research_state['iteration']}\n"
            metadata += f"- Total sources consulted: {len(self.citations)}\n"
            metadata += f"- Unique URLs accessed: {len(set(self.research_state['sources']))}\n"
            metadata += f"- Total tokens used: {self.total_tokens}\n"
            metadata += f"- Model: {self.model}\n"
            metadata += f"- Plugin version: TTD-DR Implementation v1.0\n"
            
            return report_content + references + metadata
            
        except Exception as e:
            return f"Report generation failed: {str(e)}"
    
    def research(self, system_prompt: str, initial_query: str) -> Tuple[str, int]:
        """
        Main research loop implementing TTD-DR algorithm
        
        This method orchestrates the entire research process:
        1. Query decomposition
        2. Iterative search and synthesis
        3. Completeness evaluation
        4. Focused refinement
        5. Final report generation
        """
        # Initialize research state
        self.research_state["queries"] = [initial_query]
        current_synthesis = ""
        
        for iteration in range(self.max_iterations):
            self.research_state["iteration"] = iteration + 1
            
            # Step 1: Decompose current queries (first iteration) or use focused queries
            if iteration == 0:
                queries = self.decompose_query(system_prompt, initial_query)
            else:
                # Use queries from previous iteration's gap analysis
                queries = self.research_state["queries"]
            
            # Step 2: Perform web search
            search_results = self.perform_web_search(queries)
            
            # Step 3: Extract and fetch content from URLs
            content_with_urls, sources = self.extract_and_fetch_urls(search_results)
            
            # Step 4: Synthesize information using memory plugin
            current_synthesis, tokens = self.synthesize_with_memory(
                system_prompt, initial_query, content_with_urls, sources
            )
            self.total_tokens += tokens
            
            # Step 5: Evaluate completeness
            is_complete, missing_aspects = self.evaluate_completeness(
                system_prompt, initial_query, current_synthesis
            )
            
            # Store current state
            self.research_state["content"].append(content_with_urls)
            self.research_state["synthesis"] = current_synthesis
            self.research_state["sources"].extend([s['url'] for s in sources if 'url' in s])
            
            # Check if research is complete or max iterations reached
            if is_complete or iteration == self.max_iterations - 1:
                break
            
            # Step 6: Generate focused queries for next iteration
            if missing_aspects:
                self.research_state["queries"] = self.generate_focused_queries(
                    missing_aspects, initial_query
                )
            else:
                break
        
        # Generate final structured report
        final_report = self.generate_structured_report(system_prompt, initial_query, current_synthesis)
        
        return final_report, self.total_tokens