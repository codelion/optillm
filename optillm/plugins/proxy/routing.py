"""
Routing strategies for load balancing across providers.
"""
import random
from typing import List, Optional
from abc import ABC, abstractmethod

class Router(ABC):
    """Abstract base class for routing strategies"""
    
    @abstractmethod
    def select(self, providers: List) -> Optional:
        """Select a provider from the list"""
        pass

class RoundRobinRouter(Router):
    """Round-robin routing strategy"""
    
    def __init__(self, providers: List):
        self.providers = providers
        self.index = 0
    
    def select(self, providers: List) -> Optional:
        if not providers:
            return None
        
        # Find next available provider
        for _ in range(len(providers)):
            provider = self.providers[self.index % len(self.providers)]
            self.index += 1
            if provider in providers:
                return provider
        
        return providers[0] if providers else None

class WeightedRouter(Router):
    """Weighted random routing strategy"""
    
    def __init__(self, providers: List):
        self.providers = providers
    
    def select(self, providers: List) -> Optional:
        if not providers:
            return None
        
        # Calculate weights
        weights = [p.weight for p in providers]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return random.choice(providers)
        
        # Weighted random selection
        rand = random.uniform(0, total_weight)
        cumulative = 0
        
        for provider, weight in zip(providers, weights):
            cumulative += weight
            if rand <= cumulative:
                return provider
        
        return providers[-1]

class FailoverRouter(Router):
    """Failover routing - always use first healthy provider"""
    
    def __init__(self, providers: List):
        self.providers = providers
    
    def select(self, providers: List) -> Optional:
        # Return first available provider
        return providers[0] if providers else None

class RouterFactory:
    """Factory for creating routers"""
    
    @staticmethod
    def create(strategy: str, providers: List) -> Router:
        strategies = {
            'round_robin': RoundRobinRouter,
            'weighted': WeightedRouter,
            'failover': FailoverRouter
        }
        
        router_class = strategies.get(strategy, RoundRobinRouter)
        return router_class(providers)