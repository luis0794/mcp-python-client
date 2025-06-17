# mcp-python-client

## Dependency installation

Run the following command to install the required dependencies:

```bash
uv add mcp python-dotenv google-genai
```

## How to start the MCP client

Run the following command (adjust the path according to your structure):

```bash
uv run client.py ../../servers/mcp-python-server/terminal_server.py
```

## Usage example

When the client is running, you can enter a query like:

```
create a file mcp_client_success.txt and add the text "I successfully created an MCP client with Gemini API and connected it to my MCP server"
```

## Useful resources

- [Gemini SDK for Python and other languages (Vertex AI)](https://cloud.google.com/vertex-ai/generative-ai/docs/sdks/overview?hl=es-419)
