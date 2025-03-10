"""
MCP Plugin for OptILLM

This plugin integrates the Model Context Protocol (MCP) with OptILLM,
allowing access to external tools, resources, and prompts through MCP servers.
"""

import os
import json
import logging
import asyncio
import sys
import time
import re
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from dataclasses import dataclass
from pathlib import Path
import traceback

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import mcp.types as types

# Configure logging
LOG_DIR = Path.home() / ".optillm" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "mcp_plugin.log"

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("optillm.mcp_plugin")

# Plugin identifier
SLUG = "mcp"

@dataclass
class ServerConfig:
    """Configuration for a single MCP server"""
    command: str
    args: List[str]
    env: Dict[str, str]
    description: Optional[str] = None
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'ServerConfig':
        """Create ServerConfig from a dictionary"""
        return cls(
            command=config.get("command", ""),
            args=config.get("args", []),
            env=config.get("env", {}),
            description=config.get("description")
        )

class MCPConfigManager:
    """Manages MCP configuration loading and validation"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize with optional custom config path"""
        if config_path:
            self.config_path = Path(config_path)
        else:
            self.config_path = Path.home() / ".optillm" / "mcp_config.json"
        
        # Default configuration
        self.servers: Dict[str, ServerConfig] = {}
        self.log_level: str = "INFO"
    
    def load_config(self) -> bool:
        """Load configuration from file"""
        try:
            if not self.config_path.exists():
                logger.warning(f"MCP config file not found at {self.config_path}")
                return False
                
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            # Set log level
            self.log_level = config.get("log_level", "INFO")
            log_level = getattr(logging, self.log_level.upper(), logging.INFO)
            logger.setLevel(log_level)
            
            # Load server configurations
            servers_config = config.get("mcpServers", {})
            for server_name, server_config in servers_config.items():
                self.servers[server_name] = ServerConfig.from_dict(server_config)
            
            logger.info(f"Loaded configuration with {len(self.servers)} servers")
            return True
            
        except Exception as e:
            logger.error(f"Error loading MCP configuration: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def create_default_config(self) -> bool:
        """Create a default configuration file if none exists"""
        try:
            if self.config_path.exists():
                return True
                
            default_config = {
                "mcpServers": {},
                "log_level": "INFO"
            }
            
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
                
            logger.info(f"Created default configuration at {self.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating default configuration: {e}")
            return False

class MCPServer:
    """Represents a connection to an MCP server"""
    
    def __init__(self, server_name: str, config: ServerConfig):
        self.server_name = server_name
        self.config = config
        self.session: Optional[ClientSession] = None
        self.transport: Optional[Tuple] = None
        self.connected = False
        self.tools: List[types.Tool] = []
        self.resources: List[types.Resource] = []
        self.prompts: List[types.Prompt] = []
    
    async def connect(self) -> bool:
        """Connect to the MCP server"""
        try:
            logger.info(f"Connecting to MCP server: {self.server_name}")
            
            # Create server parameters
            server_params = StdioServerParameters(
                command=self.config.command,
                args=self.config.args,
                env=self.config.env
            )
            
            # Create transport using async with
            transport = None
            try:
                # Using context manager in a way that's compatible with asyncio
                ctx = stdio_client(server_params)
                transport = await ctx.__aenter__()
                self.transport = transport
                
                read_stream, write_stream = transport
                
                # Create session
                self.session = ClientSession(read_stream, write_stream)
                
                # Initialize session
                await self.session.initialize()
                
                # Discover capabilities
                await self.discover_capabilities()
                
                self.connected = True
                logger.info(f"Successfully connected to MCP server: {self.server_name}")
                return True
                
            except Exception as e:
                # Make sure to clean up resources in case of an error
                if transport:
                    try:
                        await ctx.__aexit__(type(e), e, e.__traceback__)
                    except:
                        pass
                raise
            
        except Exception as e:
            logger.error(f"Error connecting to MCP server {self.server_name}: {e}")
            logger.error(traceback.format_exc())
            
            if self.session:
                try:
                    await self.session.aclose()
                except:
                    pass
            
            self.session = None
            self.connected = False
            return False
    
    async def discover_capabilities(self) -> bool:
        """Discover the server's capabilities"""
        if not self.session:
            logger.error(f"Cannot discover capabilities for {self.server_name}: Not connected")
            return False
            
        try:
            # List tools
            tools_result = await self.session.list_tools()
            self.tools = tools_result.tools
            
            # List resources
            resources_result = await self.session.list_resources()
            self.resources = resources_result.resources
            
            # List prompts
            prompts_result = await self.session.list_prompts()
            self.prompts = prompts_result.prompts
            
            logger.info(f"Server {self.server_name} capabilities: "
                       f"{len(self.tools)} tools, {len(self.resources)} resources, "
                       f"{len(self.prompts)} prompts")
            return True
            
        except Exception as e:
            logger.error(f"Error discovering capabilities for {self.server_name}: {e}")
            logger.error(traceback.format_exc())
            return False
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on this server"""
        if not self.session or not self.connected:
            logger.error(f"Cannot call tool for {self.server_name}: Not connected")
            return {"error": f"Server {self.server_name} is not connected"}
            
        try:
            # Find the tool
            tool = next((t for t in self.tools if t.name == tool_name), None)
            if not tool:
                return {"error": f"Tool {tool_name} not found on server {self.server_name}"}
                
            # Call the tool
            logger.info(f"Calling tool {tool_name} on server {self.server_name} with arguments: {arguments}")
            result = await self.session.call_tool(tool_name, arguments)
            
            # Process the result
            content_results = []
            for content in result.content:
                if content.type == "text":
                    content_results.append({
                        "type": "text",
                        "text": content.text
                    })
                elif content.type == "image":
                    content_results.append({
                        "type": "image",
                        "data": content.data,
                        "mimeType": content.mimeType
                    })
            
            return {
                "result": content_results,
                "is_error": result.isError
            }
            
        except Exception as e:
            logger.error(f"Error calling tool {tool_name} on server {self.server_name}: {e}")
            logger.error(traceback.format_exc())
            return {"error": f"Error calling tool: {str(e)}"}
    
    async def read_resource(self, uri: str) -> Dict[str, Any]:
        """Read a resource from this server"""
        if not self.session or not self.connected:
            logger.error(f"Cannot read resource for {self.server_name}: Not connected")
            return {"error": f"Server {self.server_name} is not connected"}
            
        try:
            # Find the resource
            resource = next((r for r in self.resources if r.uri == uri), None)
            if not resource:
                return {"error": f"Resource {uri} not found on server {self.server_name}"}
                
            # Read the resource
            logger.info(f"Reading resource {uri} from server {self.server_name}")
            result = await self.session.read_resource(uri)
            
            # Process the result
            content_results = []
            for content in result.contents:
                if content.text:
                    content_results.append({
                        "type": "text",
                        "uri": content.uri or uri,
                        "text": content.text,
                        "mimeType": content.mimeType
                    })
                elif content.blob:
                    content_results.append({
                        "type": "binary",
                        "uri": content.uri or uri,
                        "data": content.blob,
                        "mimeType": content.mimeType
                    })
            
            return {
                "result": content_results
            }
            
        except Exception as e:
            logger.error(f"Error reading resource {uri} from server {self.server_name}: {e}")
            logger.error(traceback.format_exc())
            return {"error": f"Error reading resource: {str(e)}"}
    
    async def get_prompt(self, prompt_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Get a prompt from this server"""
        if not self.session or not self.connected:
            logger.error(f"Cannot get prompt for {self.server_name}: Not connected")
            return {"error": f"Server {self.server_name} is not connected"}
            
        try:
            # Find the prompt
            prompt = next((p for p in self.prompts if p.name == prompt_name), None)
            if not prompt:
                return {"error": f"Prompt {prompt_name} not found on server {self.server_name}"}
                
            # Get the prompt
            logger.info(f"Getting prompt {prompt_name} from server {self.server_name} with arguments: {arguments}")
            result = await self.session.get_prompt(prompt_name, arguments)
            
            # Process the result
            messages = []
            for msg in result.messages:
                if msg.content.type == "text":
                    messages.append({
                        "role": msg.role,
                        "content": msg.content.text
                    })
                elif msg.content.type == "image":
                    messages.append({
                        "role": msg.role,
                        "content": {
                            "type": "image",
                            "data": msg.content.data,
                            "mimeType": msg.content.mimeType
                        }
                    })
            
            return {
                "result": messages
            }
            
        except Exception as e:
            logger.error(f"Error getting prompt {prompt_name} from server {self.server_name}: {e}")
            logger.error(traceback.format_exc())
            return {"error": f"Error getting prompt: {str(e)}"}
    
    async def close(self):
        """Close the connection to the server"""
        if self.session:
            try:
                await self.session.aclose()
                logger.info(f"Closed connection to MCP server: {self.server_name}")
            except Exception as e:
                logger.error(f"Error closing connection to {self.server_name}: {e}")
            finally:
                self.session = None
                self.connected = False

class MCPServerManager:
    """Manages MCP server connections and capabilities"""
    
    def __init__(self, config_manager: MCPConfigManager):
        self.config_manager = config_manager
        self.servers: Dict[str, MCPServer] = {}
        self.initialized = False
    
    async def initialize(self) -> bool:
        """Initialize connections to all configured servers"""
        if self.initialized:
            return True
            
        # Create servers
        for server_name, server_config in self.config_manager.servers.items():
            self.servers[server_name] = MCPServer(server_name, server_config)
        
        # Connect to all servers asynchronously
        if self.servers:
            connect_tasks = [server.connect() for server in self.servers.values()]
            results = await asyncio.gather(*connect_tasks, return_exceptions=True)
            
            # Check how many servers connected successfully
            success_count = sum(1 for r in results if r is True)
            logger.info(f"Connected to {success_count}/{len(self.servers)} MCP servers")
            
            if success_count > 0:
                self.initialized = True
                return True
            else:
                logger.error("Failed to connect to any MCP servers")
                return False
        else:
            logger.warning("No MCP servers configured")
            self.initialized = True
            return True
    
    def get_tools_for_model(self) -> List[Dict[str, Any]]:
        """Get tools from all servers in a format suitable for the model's tool-calling API"""
        tools = []
        
        for server_name, server in self.servers.items():
            if not server.connected or not server.tools:
                continue
                
            for tool in server.tools:
                # Convert MCP tool to model tool format
                tool_entry = {
                    "type": "function",
                    "function": {
                        "name": f"{server_name}.{tool.name}",
                        "description": tool.description or f"Tool {tool.name} from server {server_name}",
                        "parameters": tool.inputSchema
                    }
                }
                tools.append(tool_entry)
        
        return tools
    
    def get_capabilities_description(self) -> str:
        """Get a formatted description of all server capabilities"""
        if not self.servers:
            return "No MCP servers available."
            
        description_parts = []
        
        for server_name, server in self.servers.items():
            if not server.connected:
                description_parts.append(f"## {server_name}\nServer connection failed or not established.\n")
                continue
                
            server_description = f"## {server_name}\n"
            
            if server.config.description:
                server_description += f"{server.config.description}\n\n"
                
            if server.tools:
                server_description += "### Tools\n"
                for tool in server.tools:
                    server_description += f"- {server_name}.{tool.name}: {tool.description or 'No description'}\n"
                server_description += "\n"
                
            if server.resources:
                server_description += "### Resources\n"
                for resource in server.resources:
                    server_description += f"- {resource.uri}: {resource.name or 'No name'} - {resource.description or 'No description'}\n"
                server_description += "\n"
                
            if server.prompts:
                server_description += "### Prompts\n"
                for prompt in server.prompts:
                    server_description += f"- {prompt.name}: {prompt.description or 'No description'}\n"
                server_description += "\n"
                
            description_parts.append(server_description)
            
        return "\n".join(description_parts)
    
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool on the appropriate server"""
        if "." not in tool_name:
            return {"error": f"Invalid tool name format: {tool_name}. Expected format: server_name.tool_name"}
            
        server_name, function_name = tool_name.split(".", 1)
        
        if server_name not in self.servers:
            return {"error": f"Server not found: {server_name}"}
            
        server = self.servers[server_name]
        if not server.connected:
            return {"error": f"Server {server_name} is not connected"}
            
        # Execute the tool
        return await server.call_tool(function_name, arguments)
    
    async def close(self):
        """Close all server connections"""
        if not self.servers:
            return
            
        # Close all server connections in parallel
        close_tasks = [server.close() for server in self.servers.values()]
        await asyncio.gather(*close_tasks, return_exceptions=True)
        
        self.servers = {}
        self.initialized = False

async def run(system_prompt: str, initial_query: str, client, model: str) -> Tuple[str, int]:
    """
    Main plugin execution function called by OptILLM
    
    Args:
        system_prompt: System prompt
        initial_query: User query
        client: OptILLM client
        model: Model identifier
        
    Returns:
        Tuple of (response text, token usage)
    """
    logger.info(f"MCP Plugin run called with model: {model}")
    
    # Create server manager
    config_manager = MCPConfigManager()
    server_manager = MCPServerManager(config_manager)
    
    try:
        # Load configuration
        if not config_manager.load_config():
            # Try to create default config
            config_manager.create_default_config()
            # Try loading again
            if not config_manager.load_config():
                logger.error("Failed to load or create MCP configuration")
                # In case of no configuration, pass through the original query
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": initial_query}
                    ],
                    temperature=0.7,
                )
                return response.choices[0].message.content, response.usage.completion_tokens
        
        # Initialize server manager
        await server_manager.initialize()
        
        # Get tools formatted for the model
        tools = server_manager.get_tools_for_model()
        
        # Get capabilities description
        capabilities_description = server_manager.get_capabilities_description()
        
        # Enhance system prompt with MCP capabilities
        enhanced_system_prompt = f"{system_prompt}\n\nYou have access to the following MCP capabilities:\n\n{capabilities_description}"
        
        # First request - ask the model what it wants to do
        logger.info("Sending initial request to model")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": enhanced_system_prompt},
                {"role": "user", "content": initial_query}
            ],
            tools=tools if tools else None,  # Only include tools if available
            temperature=0.7,
        )
        
        # Check if the model wants to use any tools
        response_message = response.choices[0].message
        response_content = response_message.content or ""
        
        # Check for tool calls
        if hasattr(response_message, "tool_calls") and response_message.tool_calls:
            logger.info(f"Model requested tool calls: {len(response_message.tool_calls)}")
            
            # Create new messages with the original system and user message
            messages = [
                {"role": "system", "content": enhanced_system_prompt},
                {"role": "user", "content": initial_query},
                {"role": "assistant", "content": response_content, "tool_calls": response_message.tool_calls}
            ]
            
            # Process each tool call
            for tool_call in response_message.tool_calls:
                tool_call_id = tool_call.id
                tool_name = tool_call.function.name
                try:
                    # Parse arguments
                    arguments = json.loads(tool_call.function.arguments)
                    
                    # Execute tool
                    logger.info(f"Executing tool: {tool_name} with arguments: {arguments}")
                    result = await server_manager.execute_tool(tool_name, arguments)
                    
                    # Add tool result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": json.dumps(result)
                    })
                except Exception as e:
                    logger.error(f"Error processing tool call {tool_name}: {e}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": json.dumps({"error": f"Error: {str(e)}"})
                    })
            
            # Send follow-up request with tool results
            logger.info("Sending follow-up request to model with tool results")
            final_response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools if tools else None,  # Keep tools available in case the model wants to make additional calls
                temperature=0.7,
            )
            
            final_message = final_response.choices[0].message
            response_text = final_message.content or ""
            token_usage = final_response.usage.completion_tokens
        else:
            # Model didn't call any tools, use its initial response
            response_text = response_content
            token_usage = response.usage.completion_tokens
        
        return response_text, token_usage
        
    except Exception as e:
        logger.error(f"Error in MCP plugin run: {e}")
        logger.error(traceback.format_exc())
        # In case of error, pass through the original query
        return initial_query, 0
        
    finally:
        # Always clean up server connections
        try:
            await server_manager.close()
        except Exception as e:
            logger.error(f"Error cleaning up server connections: {e}")
