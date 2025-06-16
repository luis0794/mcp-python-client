import asyncio    # For handling asynchronous operations
import os         # For environment variable access
import sys        # For system-specific parameters and functions
import json       # For handling JSON data (used when printing function declarations)

# Import MCP client components
from typing import Optional # For type hints optional values
from contextlib import AsyncExitStack   # For managing async cleanup
from mcp import ClientSession, StdioServerParameters   # MCP session management
from mcp.client.stdio import stdio_client   # Standard I/O client for MCP

# Import Google's Gen AI SDK
from google import genai # For interacting with Google's Gen AI API
from google.genai import types # Helps structure the IA response
from google.genai.types import Tool, FunctionDeclaration # For tool and function declarations
from google.genai.types import GenerateContentConfig # For generating content configuration

from dotenv import load_dotenv  # For loading API keys from a .env file


# Load environment variables from .env file
load_dotenv()

class MCPClient:
    def __init__(self):
        """Initialize the MCP client and configure the Gemini API."""
        self.session: Optional[ClientSession] = None  # MCP session for communication
        self.exit_stack = AsyncExitStack()  # Manages async resource cleanup

        # Retrieve the Gemini API key from environment variables
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found. Please add it to your .env file.")

        # Configure the Gemini AI client
        self.genai_client = genai.Client(api_key=gemini_api_key)

    async def connect_to_server(self, server_script_path: str):
        """Connect to the MCP server and list available tools."""

        # Determine whether the server script is written in Python or JavaScript
        # This allows us to execute the correct command to start the MCP server
        command = "python" if server_script_path.endswith('.py') else "node"

        # Define the parameters for connecting to the MCP server
        server_params = StdioServerParameters(command=command, args=[server_script_path])

        # Establish communication with the MCP server using standard input/output (stdio)
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))

        # Extract the read/write streams from the transport object
        self.stdio, self.write = stdio_transport

        # Initialize the MCP client session, which allows interaction with the server
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        # Send an initialization request to the MCP server
        await self.session.initialize()

        # Request the list of available tools from the MCP server
        response = await self.session.list_tools()
        tools = response.tools  # Extract the tool list from the response

        # Print a message showing the names of the tools available on the server
        print("\nConnected to server with tools:", [tool.name for tool in tools])

        # Convert MCP tools to Gemini format
        self.function_declarations = self.convert_mcp_tools_to_gemini(tools)
    
    def convert_mcp_tools_to_gemini(self, mcp_tools):
        """
        Converts MCP tool definitions to the correct format for Gemini API function calling.

        Args:
            mcp_tools (list): List of MCP tool objects with 'name', 'description', and 'inputSchema'.

        Returns:
            list: List of Gemini Tool objects with properly formatted function declarations.
        """
        gemini_tools = []

        for tool in mcp_tools:
            # Ensure inputSchema is a valid JSON schema and clean it
            parameters = self.clean_schema(tool.inputSchema)
            if not isinstance(parameters, types.Schema):
                parameters = types.Schema(**parameters)
            function_declaration = FunctionDeclaration(
                name=tool.name,
                description=tool.description,
                parameters=parameters
            )
            gemini_tool = Tool(function_declarations=[function_declaration])
            gemini_tools.append(gemini_tool)

        return gemini_tools

    def clean_schema(self, schema):
        """
        Recursively removes 'title' fields from the JSON schema.

        Args:
            schema (dict): The schema dictionary.

        Returns:
            dict: Cleaned schema without 'title' fields.
        """
        if isinstance(schema, dict):
            schema.pop("title", None)  # Remove title if present

            # Recursively clean nested properties
            if "properties" in schema and isinstance(schema["properties"], dict):
                for key in schema["properties"]:
                    schema["properties"][key] = self.clean_schema(schema["properties"][key])

        return schema
    
    async def process_query(self, query: str) -> str:
        """
        Process a user query using the Gemini API and execute tool calls if needed.

        Args:
            query (str): The user's input query.

        Returns:
            str: The response generated by the Gemini model.
        """
        # Format user input as a structured Content object for Gemini
        user_prompt_content = types.Content(
            role='user',  # Indicates that this is a user message
            parts=[types.Part.from_text(text=query)]  # Convert the text query into a Gemini-compatible format
        )

        # Send user input to Gemini AI and include available tools for function calling
        response = self.genai_client.models.generate_content(
            model='gemini-2.0-flash-001',  # Specifies which Gemini model to use
            contents=[user_prompt_content],  # Send user input to Gemini
            config=types.GenerateContentConfig(
                tools=self.function_declarations,  # Pass the list of available MCP tools to Gemini
            )
        )

        final_text = [] # Stores the final formatted response
        assistant_message_content = [] # Stores the assistant response

        # Process the response received from Gemini
        candidates = getattr(response, 'candidates', None)
        if not response or not candidates:
            return "No response from Gemini."
        for candidate in candidates:
            if not candidate or not getattr(candidate, 'content', None):
                continue
            parts = getattr(candidate.content, 'parts', None)
            if not parts or not isinstance(parts, list):
                continue
            for part in parts:
                if isinstance(part, types.Part):
                    if getattr(part, 'function_call', None):
                        function_call_part = part
                        tool_name = getattr(function_call_part.function_call, 'name', None)
                        tool_args = getattr(function_call_part.function_call, 'args', None)
                        if not tool_name:
                            continue
                        print(f"\n[Gemini requested tool call: {tool_name} with args {tool_args}]")
                        try:
                            if not self.session:
                                function_response = {"error": "Session not initialized"}
                            else:
                                result = await self.session.call_tool(tool_name, tool_args)
                                function_response = {"result": getattr(result, 'content', str(result))}
                        except Exception as e:
                            function_response = {"error": str(e)}
                        function_response_part = types.Part.from_function_response(
                            name=tool_name,
                            response=function_response
                        )
                        function_response_content = types.Content(
                            role='tool',
                            parts=[function_response_part]
                        )
                        response = self.genai_client.models.generate_content(
                            model='gemini-2.0-flash-001',
                            contents=[
                                user_prompt_content,
                                function_call_part,
                                function_response_content
                            ],
                            config=types.GenerateContentConfig(
                                tools=self.function_declarations,
                            ),
                        )
                        if response and getattr(response, 'candidates', None) and isinstance(response.candidates, list) and len(response.candidates) > 0:
                            c0 = response.candidates[0]
                            if c0 and getattr(c0, 'content', None):
                                c0_parts = getattr(c0.content, 'parts', None)
                                if c0_parts and isinstance(c0_parts, list) and len(c0_parts) > 0 and hasattr(c0_parts[0], 'text'):
                                    final_text.append(c0_parts[0].text)
                    else:
                        if hasattr(part, 'text'):
                            final_text.append(part.text)

        # Format the final response
        return "\n".join(final_text)

    async def chat_loop(self):
        """Run an interactive chat session with the user."""
        print("\nMCP Client Started! Type 'quit' to exit.")

        while True:
            query = input("\nQuery: ").strip()
            if query.lower() == 'quit':
                break

            # Process the user's query and display the response
            response = await self.process_query(query)
            print("\n" + response)

    async def cleanup(self):
        """Clean up resources and close the MCP client."""
        await self.exit_stack.aclose()
    

async def main():
    """Main function to start the MCP client."""
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        # Connect to the MCP server and start the chat loop
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        # Ensure resources are cleaned up
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())