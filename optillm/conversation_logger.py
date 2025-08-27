import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import time

logger = logging.getLogger(__name__)

# Global logger instance - will be set by optillm.py
_global_logger: Optional['ConversationLogger'] = None

@dataclass
class ConversationEntry:
    """Represents a single conversation entry being logged"""
    request_id: str
    timestamp: str
    approach: str
    model: str
    client_request: Dict[str, Any]
    provider_calls: List[Dict[str, Any]] = field(default_factory=list)
    final_response: Optional[Dict[str, Any]] = None
    total_duration_ms: Optional[int] = None
    error: Optional[str] = None
    start_time: float = field(default_factory=time.time)

class ConversationLogger:
    """
    Logger for OptiLLM conversations including all provider interactions and metadata.
    
    Logs are saved in JSONL format (one JSON object per line) with daily rotation.
    Each entry contains the full conversation including all intermediate provider calls.
    """
    
    def __init__(self, log_dir: Path, enabled: bool = False):
        self.enabled = enabled
        self.log_dir = log_dir
        self.active_entries: Dict[str, ConversationEntry] = {}
        self._lock = threading.Lock()
        
        if self.enabled:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Conversation logging enabled. Logs will be saved to: {self.log_dir}")
        else:
            logger.debug("Conversation logging disabled")
    
    def _get_log_file_path(self, timestamp: datetime = None) -> Path:
        """Get the log file path for a given timestamp (defaults to now)"""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        date_str = timestamp.strftime("%Y-%m-%d")
        return self.log_dir / f"conversations_{date_str}.jsonl"
    
    def _generate_request_id(self) -> str:
        """Generate a unique request ID"""
        return f"req_{uuid.uuid4().hex[:8]}"
    
    def start_conversation(self, 
                          client_request: Dict[str, Any], 
                          approach: str, 
                          model: str) -> str:
        """
        Start logging a new conversation.
        
        Args:
            client_request: The original request from the client
            approach: The optimization approach being used
            model: The model name
            
        Returns:
            str: Unique request ID for this conversation
        """
        if not self.enabled:
            return ""
        
        request_id = self._generate_request_id()
        timestamp = datetime.now(timezone.utc).isoformat()
        
        entry = ConversationEntry(
            request_id=request_id,
            timestamp=timestamp,
            approach=approach,
            model=model,
            client_request=client_request.copy()
        )
        
        with self._lock:
            self.active_entries[request_id] = entry
        
        logger.debug(f"Started conversation logging for request {request_id}")
        return request_id
    
    def log_provider_call(self, 
                         request_id: str, 
                         provider_request: Dict[str, Any], 
                         provider_response: Dict[str, Any]) -> None:
        """
        Log a provider API call and response.
        
        Args:
            request_id: The request ID for this conversation
            provider_request: The request sent to the provider
            provider_response: The response received from the provider
        """
        if not self.enabled or not request_id:
            return
        
        with self._lock:
            entry = self.active_entries.get(request_id)
            if not entry:
                logger.warning(f"No active conversation found for request {request_id}")
                return
            
            call_data = {
                "call_number": len(entry.provider_calls) + 1,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "request": provider_request.copy(),
                "response": provider_response.copy()
            }
            
            entry.provider_calls.append(call_data)
            
        logger.debug(f"Logged provider call #{len(entry.provider_calls)} for request {request_id}")
    
    def log_final_response(self, 
                          request_id: str, 
                          final_response: Dict[str, Any]) -> None:
        """
        Log the final response sent back to the client.
        
        Args:
            request_id: The request ID for this conversation
            final_response: The final response sent to the client
        """
        if not self.enabled or not request_id:
            return
        
        with self._lock:
            entry = self.active_entries.get(request_id)
            if not entry:
                logger.warning(f"No active conversation found for request {request_id}")
                return
            
            entry.final_response = final_response.copy()
            entry.final_response["timestamp"] = datetime.now(timezone.utc).isoformat()
    
    def log_error(self, request_id: str, error: str) -> None:
        """
        Log an error for this conversation.
        
        Args:
            request_id: The request ID for this conversation  
            error: Error message or description
        """
        if not self.enabled or not request_id:
            return
        
        with self._lock:
            entry = self.active_entries.get(request_id)
            if not entry:
                logger.warning(f"No active conversation found for request {request_id}")
                return
            
            entry.error = error
            
        logger.debug(f"Logged error for request {request_id}: {error}")
    
    def finalize_conversation(self, request_id: str) -> None:
        """
        Finalize and save the conversation to disk.
        
        Args:
            request_id: The request ID for this conversation
        """
        if not self.enabled or not request_id:
            return
        
        with self._lock:
            entry = self.active_entries.pop(request_id, None)
            if not entry:
                logger.warning(f"No active conversation found for request {request_id}")
                return
            
            # Calculate total duration
            entry.total_duration_ms = int((time.time() - entry.start_time) * 1000)
            
            # Convert to dict for JSON serialization
            log_entry = {
                "timestamp": entry.timestamp,
                "request_id": entry.request_id,
                "approach": entry.approach,
                "model": entry.model,
                "client_request": entry.client_request,
                "provider_calls": entry.provider_calls,
                "final_response": entry.final_response,
                "total_duration_ms": entry.total_duration_ms,
                "error": entry.error
            }
            
            # Write to log file
            self._write_log_entry(log_entry)
        
        logger.debug(f"Finalized conversation for request {request_id}")
    
    def _write_log_entry(self, log_entry: Dict[str, Any]) -> None:
        """Write a log entry to the appropriate JSONL file"""
        try:
            log_file_path = self._get_log_file_path()
            with open(log_file_path, 'a', encoding='utf-8') as f:
                json.dump(log_entry, f, separators=(',', ':'))
                f.write('\n')
            logger.debug(f"Wrote log entry to {log_file_path}")
        except Exception as e:
            logger.error(f"Failed to write log entry: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about conversation logging"""
        with self._lock:
            active_count = len(self.active_entries)
        
        stats = {
            "enabled": self.enabled,
            "log_dir": str(self.log_dir),
            "active_conversations": active_count
        }
        
        if self.enabled:
            # Count total log files and approximate total entries
            log_files = list(self.log_dir.glob("conversations_*.jsonl"))
            total_entries = 0
            for log_file in log_files:
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        total_entries += sum(1 for line in f if line.strip())
                except Exception:
                    pass
            
            stats.update({
                "log_files_count": len(log_files),
                "total_entries_approximate": total_entries
            })
        
        return stats


# Module-level functions for easy access from approach modules
def set_global_logger(logger_instance: 'ConversationLogger') -> None:
    """Set the global logger instance (called by optillm.py)"""
    global _global_logger
    _global_logger = logger_instance


def log_provider_call(request_id: str, provider_request: Dict[str, Any], provider_response: Dict[str, Any]) -> None:
    """Log a provider call using the global logger instance"""
    if _global_logger and _global_logger.enabled:
        _global_logger.log_provider_call(request_id, provider_request, provider_response)


def log_error(request_id: str, error_message: str) -> None:
    """Log an error using the global logger instance"""
    if _global_logger and _global_logger.enabled:
        _global_logger.log_error(request_id, error_message)