"""
ProxyClient implementation for load balancing across multiple LLM providers.
"""
import time
import logging
import random
from typing import Dict, List, Any, Optional
from openai import OpenAI, AzureOpenAI
from optillm.plugins.proxy.routing import RouterFactory
from optillm.plugins.proxy.health import HealthChecker

logger = logging.getLogger(__name__)

class Provider:
    """Wrapper for a provider configuration and client"""
    def __init__(self, config: Dict):
        self.name = config['name']
        self.base_url = config['base_url']
        self.api_key = config['api_key']
        self.weight = config.get('weight', 1)
        self.fallback_only = config.get('fallback_only', False)
        self.model_map = config.get('model_map', {})
        self._client = None
        self.is_healthy = True
        self.last_error = None
        self.latencies = []  # Track recent latencies
        
    @property
    def client(self):
        """Lazy initialization of OpenAI client"""
        if not self._client:
            if 'azure' in self.base_url.lower():
                # Handle Azure OpenAI
                self._client = AzureOpenAI(
                    api_key=self.api_key,
                    azure_endpoint=self.base_url,
                    api_version="2024-02-01"
                )
            else:
                # Standard OpenAI-compatible client
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url
                )
        return self._client
    
    def map_model(self, model: str) -> str:
        """Map requested model to provider-specific name"""
        return self.model_map.get(model, model)
    
    def track_latency(self, latency: float):
        """Track request latency"""
        self.latencies.append(latency)
        if len(self.latencies) > 10:
            self.latencies.pop(0)
    
    def avg_latency(self) -> float:
        """Get average latency"""
        if not self.latencies:
            return 0
        return sum(self.latencies) / len(self.latencies)

class ProxyClient:
    """OpenAI-compatible client that proxies to multiple providers"""
    
    def __init__(self, config: Dict, fallback_client=None):
        self.config = config
        self.fallback_client = fallback_client
        
        # Initialize providers
        self.providers = [
            Provider(p) for p in config.get('providers', [])
        ]
        
        # Filter out fallback-only providers for normal routing
        self.active_providers = [
            p for p in self.providers if not p.fallback_only
        ]
        
        self.fallback_providers = [
            p for p in self.providers if p.fallback_only
        ]
        
        # Initialize router
        strategy = config.get('routing', {}).get('strategy', 'round_robin')
        self.router = RouterFactory.create(strategy, self.active_providers)
        
        # Initialize health checker
        health_config = config.get('routing', {}).get('health_check', {})
        self.health_checker = HealthChecker(
            providers=self.providers,
            enabled=health_config.get('enabled', True),
            interval=health_config.get('interval', 30),
            timeout=health_config.get('timeout', 5)
        )
        
        # Start health checking
        self.health_checker.start()
        
        # Monitoring settings
        monitoring = config.get('monitoring', {})
        self.track_latency = monitoring.get('track_latency', True)
        self.track_errors = monitoring.get('track_errors', True)
        
        # Create chat namespace
        self.chat = self._Chat(self)
    
    class _Chat:
        def __init__(self, proxy_client):
            self.proxy_client = proxy_client
            self.completions = proxy_client._Completions(proxy_client)
    
    class _Completions:
        def __init__(self, proxy_client):
            self.proxy_client = proxy_client
        
        def _filter_kwargs(self, kwargs: dict) -> dict:
            """Filter out OptiLLM-specific parameters that shouldn't be sent to providers"""
            optillm_params = {
                'optillm_approach', 'proxy_wrap', 'wrapped_approach', 'wrap',
                'mcts_simulations', 'mcts_exploration', 'mcts_depth',
                'best_of_n', 'rstar_max_depth', 'rstar_num_rollouts', 'rstar_c'
            }
            return {k: v for k, v in kwargs.items() if k not in optillm_params}
        
        def create(self, **kwargs):
            """Create completion with load balancing and failover"""
            model = kwargs.get('model', 'unknown')
            attempted_providers = set()
            errors = []
            
            # Get healthy providers
            healthy_providers = [
                p for p in self.proxy_client.active_providers 
                if p.is_healthy
            ]
            
            if not healthy_providers:
                logger.warning("No healthy providers, trying fallback providers")
                healthy_providers = self.proxy_client.fallback_providers
            
            # Try routing through healthy providers
            while healthy_providers:
                available_providers = [p for p in healthy_providers if p not in attempted_providers]
                if not available_providers:
                    break
                    
                provider = self.proxy_client.router.select(available_providers)
                logger.info(f"Router selected provider: {provider.name if provider else 'None'}")
                
                if not provider:
                    break
                    
                attempted_providers.add(provider)
                
                try:
                    # Map model name if needed and filter out OptiLLM-specific parameters
                    request_kwargs = self._filter_kwargs(kwargs.copy())
                    request_kwargs['model'] = provider.map_model(model)
                    
                    # Track timing
                    start_time = time.time()
                    
                    # Make request
                    logger.debug(f"Routing to {provider.name}")
                    response = provider.client.chat.completions.create(**request_kwargs)
                    
                    # Track success
                    latency = time.time() - start_time
                    if self.proxy_client.track_latency:
                        provider.track_latency(latency)
                    
                    logger.info(f"Request succeeded via {provider.name} in {latency:.2f}s")
                    return response
                    
                except Exception as e:
                    logger.error(f"Provider {provider.name} failed: {e}")
                    errors.append((provider.name, str(e)))
                    
                    # Mark provider as unhealthy
                    if self.proxy_client.track_errors:
                        provider.is_healthy = False
                        provider.last_error = str(e)
            
            # All providers failed, try fallback client
            if self.proxy_client.fallback_client:
                logger.warning("All proxy providers failed, using fallback client")
                try:
                    return self.proxy_client.fallback_client.chat.completions.create(**self._filter_kwargs(kwargs))
                except Exception as e:
                    errors.append(("fallback_client", str(e)))
            
            # Complete failure
            error_msg = f"All providers failed. Errors: {errors}"
            logger.error(error_msg)
            raise Exception(error_msg)