import os
import json
import logging
import asyncio
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum
import re
import pydantic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import mcp.types as types

logger = logging.getLogger(__name__)

# Plugin identifier
SLUG = "mcp"

class ServerType(str, Enum):
    """Supported server types"""
    STDIO = "stdio"
    HTTP = "http"

class ResourceAccess(str, Enum):
    """Resource access modes"""
    READ_ONLY = "read_only"
    READ_WRITE = "read_write"
    NONE = "none"

class ServerConfig(pydantic.BaseModel):
    """Configuration for a single MCP server"""
    type: ServerType = ServerType.STDIO
    command: str
    args: List[str] = []
    env: Dict[str, str] = {}
    url: Optional[str] = None  # For HTTP servers
    resource_access: ResourceAccess = ResourceAccess.READ_ONLY
    allowed_tools: Optional[List[str]] = None  # If None, all tools allowed
    description: Optional[str] = None

class MCPConfig(pydantic.BaseModel):
    """Root configuration model"""
    mcpServers: Dict[str, ServerConfig]
    default_model: Optional[str] = None
    log_level: str = "INFO"

@dataclass
class ToolMatch:
    """Represents a matched tool with extracted arguments"""
    server_name: str
    tool_name: str
    arguments: Dict[str, Any]
    confidence: float
    
@dataclass
class PromptMatch:
    """Represents a matched prompt template"""
    server_name: str
    prompt_name: str
    arguments: Dict[str, Any]

class ArgumentExtractor:
    """Extracts arguments from text using LLM"""
    
    def __init__(self, client, model: str):
        self.client = client
        self.model = model

    async def extract_arguments(self, text: str, tool: types.Tool) -> Dict[str, Any]:
        """Use LLM to extract arguments from text"""
        prompt = f"""
        Extract arguments for the tool '{tool.name}' from this text: "{text}"
        
        The tool accepts these arguments:
        {json.dumps(tool.inputSchema, indent=2)}
        
        Return only a JSON object with the extracted arguments. If an argument can't be found, omit it.
        """
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        
        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            logger.error("Failed to parse argument extraction response")
            return {}

class ToolMatcher:
    """Matches tools to text using semantic and pattern matching"""
    
    def __init__(self, client, model: str):
        self.client = client
        self.model = model
        self.argument_extractor = ArgumentExtractor(client, model)

    async def find_matching_tools(
        self, 
        text: str, 
        available_tools: Dict[str, List[types.Tool]]
    ) -> List[ToolMatch]:
        """Find tools that match the given text"""
        matches = []
        
        # First pass: Look for explicit tool mentions
        for server_name, tools in available_tools.items():
            for tool in tools:
                # Check for direct name matches
                if tool.name.lower() in text.lower():
                    args = await self.argument_extractor.extract_arguments(text, tool)
                    matches.append(ToolMatch(
                        server_name=server_name,
                        tool_name=tool.name,
                        arguments=args,
                        confidence=0.9
                    ))
                    continue
                
                # Check for semantic matches using the tool description
                if tool.description and self._semantic_match(text, tool.description):
                    args = await self.argument_extractor.extract_arguments(text, tool)
                    matches.append(ToolMatch(
                        server_name=server_name,
                        tool_name=tool.name,
                        arguments=args,
                        confidence=0.7
                    ))
        
        # Use LLM for additional tool matching if needed
        if not matches:
            llm_matches = await self._llm_tool_matching(text, available_tools)
            matches.extend(llm_matches)
        
        # Sort by confidence
        matches.sort(key=lambda x: x.confidence, reverse=True)
        return matches

    def _semantic_match(self, text: str, description: str) -> bool:
        """Simple semantic matching using keywords"""
        keywords = description.lower().split()
        text_words = text.lower().split()
        matches = sum(1 for word in keywords if any(w.startswith(word) for w in text_words))
        return matches / len(keywords) > 0.5

    async def _llm_tool_matching(
        self, 
        text: str, 
        available_tools: Dict[str, List[types.Tool]]
    ) -> List[ToolMatch]:
        """Use LLM to find matching tools"""
        # Create tool descriptions
        tool_descriptions = []
        for server_name, tools in available_tools.items():
            for tool in tools:
                desc = f"Server: {server_name}, Tool: {tool.name}"
                if tool.description:
                    desc += f", Description: {tool.description}"
                tool_descriptions.append(desc)
        
        prompt = f"""
        Given this user request: "{text}"
        
        And these available tools:
        {json.dumps(tool_descriptions, indent=2)}
        
        Which tools would be most appropriate to use? Return a JSON array of objects with:
        - server_name: The server name
        - tool_name: The tool name
        - confidence: A number between 0 and 1 indicating confidence
        
        Only include tools with confidence > 0.5.
        """
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        
        try:
            llm_matches = json.loads(response.choices[0].message.content)
            matches = []
            
            for match in llm_matches:
                server_name = match["server_name"]
                tool_name = match["tool_name"]
                
                # Find the tool
                tool = next(
                    (t for t in available_tools[server_name] if t.name == tool_name),
                    None
                )
                
                if tool:
                    args = await self.argument_extractor.extract_arguments(text, tool)
                    matches.append(ToolMatch(
                        server_name=server_name,
                        tool_name=tool_name,
                        arguments=args,
                        confidence=match["confidence"]
                    ))
            
            return matches
        except (json.JSONDecodeError, KeyError):
            logger.error("Failed to parse LLM tool matching response")
            return []

class ResourceManager:
    """Manages MCP resources"""
    
    def __init__(self, client, model: str):
        self.client = client
        self.model = model
        self.resource_cache: Dict[str, List[types.Resource]] = {}

    async def find_relevant_resources(
        self, 
        text: str, 
        available_resources: Dict[str, List[types.Resource]]
    ) -> List[str]:
        """Find resources relevant to the given text"""
        prompt = f"""
        Given this user request: "{text}"
        
        And these available resources:
        {json.dumps([
            {
                "server": server,
                "resources": [
                    {"uri": r.uri, "name": r.name, "description": r.description}
                    for r in resources
                ]
            }
            for server, resources in available_resources.items()
        ], indent=2)}
        
        Which resources would be most relevant? Return only a JSON array of resource URIs.
        """
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        
        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            logger.error("Failed to parse resource relevance response")
            return []

class MCPClientManager:
    """Manages multiple MCP client connections"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or os.path.expanduser("~/mcp_config.json")
        self.sessions: Dict[str, ClientSession] = {}
        self.tools_cache: Dict[str, List[types.Tool]] = {}
        self.resources_cache: Dict[str, List[types.Resource]] = {}
        self.prompts_cache: Dict[str, List[types.Prompt]] = {}
        self.config: Optional[MCPConfig] = None
        
    def validate_config(self, config_data: Dict[str, Any]) -> MCPConfig:
        """Validate configuration using pydantic"""
        try:
            return MCPConfig(**config_data)
        except pydantic.ValidationError as e:
            logger.error(f"Invalid configuration: {e}")
            raise
            
    async def initialize_servers(self):
        """Initialize connections to all configured servers"""
        try:
            with open(self.config_path, 'r') as f:
                config_data = json.load(f)
            
            self.config = self.validate_config(config_data)
            
            for server_name, server_config in self.config.mcpServers.items():
                try:
                    await self.connect_server(server_name, server_config)
                except Exception as e:
                    logger.error(f"Failed to connect to server {server_name}: {e}")
                    
        except FileNotFoundError:
            logger.warning(f"MCP config file not found at {self.config_path}")
        except Exception as e:
            logger.error(f"Error initializing MCP servers: {e}")

    async def connect_server(self, server_name: str, server_config: ServerConfig):
        """Connect to a single MCP server"""
        if server_config.type == ServerType.STDIO:
            server_params = StdioServerParameters(
                command=server_config.command,
                args=server_config.args,
                env=server_config.env
            )
            
            transport = await stdio_client(server_params)
            read_stream, write_stream = transport
            
        elif server_config.type == ServerType.HTTP:
            # HTTP transport implementation would go here
            raise NotImplementedError("HTTP transport not yet implemented")
            
        try:
            session = ClientSession(read_stream, write_stream)
            await session.initialize()
            
            # Cache available capabilities
            tools_result = await session.list_tools()
            self.tools_cache[server_name] = tools_result.tools
            
            resources_result = await session.list_resources()
            self.resources_cache[server_name] = resources_result.resources
            
            prompts_result = await session.list_prompts()
            self.prompts_cache[server_name] = prompts_result.prompts
            
            self.sessions[server_name] = session
            logger.info(f"Connected to MCP server: {server_name}")
            
        except Exception as e:
            logger.error(f"Error connecting to server {server_name}: {e}")
            raise

    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Call a tool on a specific server"""
        if server_name not in self.sessions:
            raise ValueError(f"Server {server_name} not connected")
            
        # Validate against allowed tools if configured
        server_config = self.config.mcpServers[server_name]
        if server_config.allowed_tools is not None:
            if tool_name not in server_config.allowed_tools:
                raise ValueError(f"Tool {tool_name} not allowed on server {server_name}")
        
        session = self.sessions[server_name]
        result = await session.call_tool(tool_name, arguments)
        
        # Extract text content from result
        text_contents = []
        for content in result.content:
            if content.type == "text":
                text_contents.append(content.text)
        
        return "\n".join(text_contents)

    async def read_resource(self, server_name: str, uri: str) -> Tuple[str, Optional[str]]:
        """Read a resource from a server"""
        if server_name not in self.sessions:
            raise ValueError(f"Server {server_name} not connected")
            
        # Check resource access permissions
        server_config = self.config.mcpServers[server_name]
        if server_config.resource_access == ResourceAccess.NONE:
            raise ValueError(f"Resource access not allowed on server {server_name}")
            
        session = self.sessions[server_name]
        result = await session.read_resource(uri)
        
        # Return first content and its MIME type
        if result.contents:
            content = result.contents[0]
            return content.text or content.blob or "", content.mimeType
        return "", None

    async def get_prompt(self, server_name: str, prompt_name: str, arguments: Dict[str, Any]) -> str:
        """Get a prompt from a server"""
        if server_name not in self.sessions:
            raise ValueError(f"Server {server_name} not connected")
            
        session = self.sessions[server_name]
        result = await session.get_prompt(prompt_name, arguments)
        
        # Convert prompt messages to text
        messages = []
        for msg in result.messages:
            if msg.content.type == "text":
                messages.append(f"{msg.role}: {msg.content.text}")
        
        return "\n".join(messages)

    async def cleanup(self):
        """Clean up all server connections"""
        for session in self.sessions.values():
            await session.aclose()
        self.sessions.clear()
        self.tools_cache.clear()
        self.resources_cache.clear()
        self.prompts_cache.clear()

class MCPPlugin:
    """optillm plugin for MCP integration"""
    
    def __init__(self):
        self.client_manager = MCPClientManager()
        self.initialized = False
        self.tool_matcher: Optional[ToolMatcher] = None
        self.resource_manager: Optional[ResourceManager] = None
        
    async def ensure_initialized(self, client, model: str):
        """Initialize if not already done"""
        if not self.initialized:
            await self.client_manager.initialize_servers()
            self.tool_matcher = ToolMatcher(client, model)
            self.resource_manager = ResourceManager(client, model)
            self.initialized = True

    async def process_request(
            self, 
            messages: List[Dict[str, Any]], 
            model: str
        ) -> str:
            """Process the request and handle MCP interactions"""
            # Last message contains the current request
            current_message = messages[-1]["content"]
            
            # Find matching tools
            tool_matches = await self.tool_matcher.find_matching_tools(
                current_message, 
                self.client_manager.tools_cache
            )
            
            # Find relevant resources
            relevant_resources = await self.resource_manager.find_relevant_resources(
                current_message,
                self.client_manager.resources_cache
            )
            
            # Collect context and results
            context_parts = []
            
            # Add resource content
            for uri in relevant_resources:
                server_name = uri.split("://")[0]  # Simple server extraction from URI
                try:
                    content, mime_type = await self.client_manager.read_resource(
                        server_name,
                        uri
                    )
                    if content:
                        context_parts.append(f"Resource {uri}:\n{content}")
                except Exception as e:
                    logger.error(f"Error reading resource {uri}: {e}")
            
            # Execute tool calls
            for match in tool_matches:
                try:
                    result = await self.client_manager.call_tool(
                        match.server_name,
                        match.tool_name,
                        match.arguments
                    )
                    context_parts.append(f"Tool {match.tool_name} result:\n{result}")
                except Exception as e:
                    logger.error(f"Error calling tool {match.tool_name}: {e}")
                    context_parts.append(f"Error calling tool {match.tool_name}: {str(e)}")
            
            # Build final context
            context = "\n\n".join(context_parts)
            if context:
                return f"{current_message}\n\nContext:\n{context}"
            return current_message

    async def handle_tool_error(
        self,
        error: Exception,
        tool_match: ToolMatch,
        client,
        model: str
    ) -> str:
        """Handle tool execution errors intelligently"""
        prompt = f"""
        An error occurred while executing tool '{tool_match.tool_name}':
        Error: {str(error)}
        
        The tool was called with these arguments:
        {json.dumps(tool_match.arguments, indent=2)}
        
        Analyze the error and provide a brief explanation of what went wrong.
        Focus on possible solutions or alternatives.
        """
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        return response.choices[0].message.content

async def run(system_prompt: str, initial_query: str, client, model: str) -> Tuple[str, int]:
    """Main plugin execution function"""
    plugin = MCPPlugin()
    
    try:
        await plugin.ensure_initialized(client, model)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": initial_query}
        ]
        
        processed_query = await plugin.process_request(messages, model)
        
        # Create a system prompt that includes MCP capabilities
        enhanced_system_prompt = f"""
        {system_prompt}
        
        You have access to the following MCP capabilities:
        
        Tools:
        {json.dumps([
            {
                "server": server,
                "tools": [
                    {"name": t.name, "description": t.description}
                    for t in tools
                ]
            }
            for server, tools in plugin.client_manager.tools_cache.items()
        ], indent=2)}
        
        Resources:
        {json.dumps([
            {
                "server": server,
                "resources": [
                    {"uri": r.uri, "name": r.name, "description": r.description}
                    for r in resources
                ]
            }
            for server, resources in plugin.client_manager.resources_cache.items()
        ], indent=2)}
        
        Prompts:
        {json.dumps([
            {
                "server": server,
                "prompts": [
                    {"name": p.name, "description": p.description}
                    for p in prompts
                ]
            }
            for server, prompts in plugin.client_manager.prompts_cache.items()
        ], indent=2)}
        """
        
        # Pass the processed query and enhanced system prompt to the model
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": enhanced_system_prompt},
                {"role": "user", "content": processed_query}
            ],
            temperature=0.7,
        )
        
        return response.choices[0].message.content, response.usage.completion_tokens
        
    except Exception as e:
        logger.error(f"Error in MCP plugin: {str(e)}")
        # In case of error, pass through the original query
        return initial_query, 0
    finally:
        await plugin.client_manager.cleanup()

def validate_config_file(config_path: str) -> None:
    """Validate MCP configuration file"""
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        MCPConfig(**config_data)
    except FileNotFoundError:
        raise ValueError(f"Configuration file not found: {config_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file: {e}")
    except pydantic.ValidationError as e:
        raise ValueError(f"Invalid configuration format: {e}")
    except Exception as e:
        raise ValueError(f"Error validating configuration: {e}")

def create_default_config(config_path: str) -> None:
    """Create a default MCP configuration file"""
    default_config = {
        "mcpServers": {
            "example": {
                "type": "stdio",
                "command": "python",
                "args": ["example_server.py"],
                "env": {},
                "resource_access": "read_only",
                "description": "Example MCP server"
            }
        },
        "log_level": "INFO"
    }
    
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(default_config, f, indent=2)

async def test_server_connection(
    server_name: str,
    server_config: ServerConfig
) -> Tuple[bool, str]:
    """Test connection to a single MCP server"""
    try:
        if server_config.type == ServerType.STDIO:
            server_params = StdioServerParameters(
                command=server_config.command,
                args=server_config.args,
                env=server_config.env
            )
            
            transport = await stdio_client(server_params)
            read_stream, write_stream = transport
            
            session = ClientSession(read_stream, write_stream)
            await session.initialize()
            
            # Test basic operations
            await session.list_tools()
            await session.list_resources()
            await session.list_prompts()
            
            await session.aclose()
            return True, "Connection successful"
            
        elif server_config.type == ServerType.HTTP:
            return False, "HTTP transport not yet implemented"
            
    except Exception as e:
        return False, f"Connection failed: {str(e)}"