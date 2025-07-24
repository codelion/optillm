import asyncio
import json
import re
from typing import Tuple, List, Dict, Optional, Any
from optillm.plugins.web_search_plugin import run as web_search_run
from optillm.plugins.readurls_plugin import run as readurls_run
from optillm.plugins.memory_plugin import run as memory_run

SLUG = "deep_research"

class DeepResearcher:
    """
    Implementation of Test-Time Diffusion Deep Researcher (TTD-DR) algorithm
    
    This class implements the paper's approach of treating research as a diffusion process
    with iterative refinement through denoising and retrieval.
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
        combined_query = "Search for the following topics:\n" + "\n".join([f"- {q}" for q in queries])
        
        try:
            enhanced_query, _ = web_search_run("", combined_query, None, None, {
                "num_results": self.max_sources,
                "delay_seconds": 3,  # Increased delay to avoid rate limiting
                "headless": True
            })
            return enhanced_query
        except Exception as e:
            return f"Web search failed: {str(e)}"
    
    def extract_and_fetch_urls(self, search_results: str) -> str:
        """
        Extract URLs from search results and fetch their content using readurls plugin
        """
        try:
            content_with_urls, _ = readurls_run("", search_results, None, None)
            return content_with_urls
        except Exception as e:
            return f"URL fetching failed: {str(e)}"
    
    def synthesize_with_memory(self, system_prompt: str, query: str, content: str) -> Tuple[str, int]:
        """
        Use memory plugin to synthesize information from collected content
        """
        # Format content for memory plugin (it expects "Query: <query>" format)
        memory_input = f"{content}\n\nQuery: {query}"
        
        try:
            synthesis, tokens = memory_run(system_prompt, memory_input, self.client, self.model)
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
    
    def research(self, system_prompt: str, initial_query: str) -> Tuple[str, int]:
        """
        Main research loop implementing TTD-DR algorithm
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
            content_with_urls = self.extract_and_fetch_urls(search_results)
            
            # Step 4: Synthesize information using memory plugin
            current_synthesis, tokens = self.synthesize_with_memory(
                system_prompt, initial_query, content_with_urls
            )
            self.total_tokens += tokens
            
            # Step 5: Evaluate completeness
            is_complete, missing_aspects = self.evaluate_completeness(
                system_prompt, initial_query, current_synthesis
            )
            
            # Store current state
            self.research_state["content"].append(content_with_urls)
            self.research_state["synthesis"] = current_synthesis
            
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
        
        # Generate final comprehensive response
        final_response = self.generate_final_response(system_prompt, initial_query, current_synthesis)
        
        return final_response, self.total_tokens
    
    def generate_final_response(self, system_prompt: str, original_query: str, synthesis: str) -> str:
        """
        Generate the final comprehensive research response
        """
        final_prompt = f"""
        Based on comprehensive research, provide a detailed and well-structured response to the following query.
        
        Original query: {original_query}
        Research synthesis: {synthesis}
        
        Please provide a comprehensive, well-organized response that:
        1. Directly addresses the original query
        2. Includes key findings and insights
        3. Provides proper context and background
        4. Is well-structured with clear sections
        5. Acknowledges any limitations or areas where more research might be needed
        
        Format your response professionally and cite specific information where relevant.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": final_prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            final_content = response.choices[0].message.content.strip()
            self.total_tokens += response.usage.completion_tokens
            
            # Add research metadata
            metadata = f"\n\n---\n**Research Summary:**\n"
            metadata += f"- Iterations completed: {self.research_state['iteration']}\n"
            metadata += f"- Total tokens used: {self.total_tokens}\n"
            metadata += f"- Sources consulted: Multiple web sources and documents\n"
            
            return final_content + metadata
            
        except Exception as e:
            return f"Final response generation failed: {str(e)}"

def run(system_prompt: str, initial_query: str, client, model: str, request_config: Optional[Dict] = None) -> Tuple[str, int]:
    """
    Deep Research plugin implementing TTD-DR (Test-Time Diffusion Deep Researcher)
    
    This plugin orchestrates web search, URL fetching, and memory synthesis to provide
    comprehensive research responses using an iterative refinement approach.
    
    Args:
        system_prompt: System prompt for the conversation
        initial_query: User's research query
        client: OpenAI client for LLM calls
        model: Model name to use for synthesis
        request_config: Optional configuration dict with keys:
            - max_iterations: Maximum research iterations (default: 5)
            - max_sources: Maximum web sources per search (default: 10)
    
    Returns:
        Tuple of (comprehensive_research_response, total_completion_tokens)
    """
    # Parse configuration
    config = request_config or {}
    max_iterations = config.get("max_iterations", 5)
    max_sources = config.get("max_sources", 10)
    
    # Validate inputs
    if not initial_query.strip():
        return "Error: No research query provided", 0
    
    if not client:
        return "Error: No LLM client provided for research synthesis", 0
    
    # Initialize researcher
    researcher = DeepResearcher(
        client=client,
        model=model,
        max_iterations=max_iterations,
        max_sources=max_sources
    )
    
    try:
        # Perform deep research
        result, total_tokens = researcher.research(system_prompt, initial_query)
        return result, total_tokens
        
    except Exception as e:
        error_response = f"Deep research failed: {str(e)}\n\nFalling back to basic response..."
        
        # Fallback: provide basic response using just the model
        try:
            fallback_response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": initial_query}
                ]
            )
            
            result = fallback_response.choices[0].message.content.strip()
            tokens = fallback_response.usage.completion_tokens
            
            return f"{error_response}\n\n{result}", tokens
            
        except Exception as fallback_error:
            return f"Deep research and fallback both failed: {str(e)} | {str(fallback_error)}", 0