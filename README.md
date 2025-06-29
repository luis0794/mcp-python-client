# mcp-python-client

A Python client for interacting with Model Context Protocol (MCP) servers, featuring integration with Google's Gemini API for natural language processing.

## Features

- Connect to multiple MCP servers simultaneously
- Interact with MCP tools using natural language
- Integration with Google's Gemini AI for intelligent tool usage
- Support for various MCP server types (terminal, memory, filesystem, fetch, etc.)

## Dependency Installation

Run the following command to install the required dependencies:

```bash
uv add mcp python-dotenv google-genai
```

## Configuration

The client uses a JSON configuration file (`langchain_multi_server_config.json`) to define and manage MCP servers. The following servers are pre-configured:

1. **Terminal Server**
   - Runs in a Docker container
   - Provides terminal/shell access in a sandboxed environment
   - Mounts a local workspace directory for file operations

2. **Memory Server**
   - Provides short-term memory capabilities
   - Useful for maintaining context across multiple interactions

3. **Filesystem Server**
   - Provides access to the local filesystem
   - Configured to access the user's home directory

4. **Fetch Server**
   - Enables web content fetching capabilities
   - Can be used to retrieve and process web content
   - Requires `mcp-server-fetch` to be installed via `uvx`

5. **Atlassian Server** (disabled by default)
   - Connects to Atlassian services
   - Requires appropriate credentials

6. **Playwright Server** (disabled by default)
   - Provides browser automation capabilities
   - Useful for web scraping and testing

## How to Start the MCP Client

### Single Server Mode

Run the client with a specific server script:

```bash
uv run client.py ../../servers/mcp-python-server/terminal_server.py
```

### Multi-Server Mode

Use the LangChain client with multiple MCP servers:

```bash
uv run langchain_mcp_client_wconfig.py
```

Make sure to set the `LANGCHAIN_MULTI_SERVER_CONFIG` environment variable to point to your configuration file if it's not in the default location.

## Usage Example

When the client is running, you can enter natural language queries like:

```
create a file mcp_client_success.txt and add the text "I successfully created an MCP client with Gemini API and connected it to my MCP server"
```

Or interact with the fetch server:

```
fetch the content from https://example.com and save it as example.html
```

## Environment Variables

- `GOOGLE_API_KEY`: Required for Google Gemini API access
- `LANGCHAIN_MULTI_SERVER_CONFIG`: Path to the MCP server configuration file (defaults to `langchain_multi_server_config.json` in the current directory)

## Useful Resources

- [Gemini SDK for Python and other languages (Vertex AI)](https://cloud.google.com/vertex-ai/generative-ai/docs/sdks/overview?hl=es-419)
- [Model Context Protocol Documentation](https://github.com/modelcontextprotocol/servers/tree/main/src/fetch)
- [MCP Server Implementations](https://github.com/modelcontextprotocol/servers)
