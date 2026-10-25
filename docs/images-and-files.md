# Images and files

A prompt can be a list instead of a string: text, images and files (such as PDFs), in any order.

```python
from pyox_agents import Agent, file, image

agent = Agent(model="gpt-5-mini")

await agent.run(["What is in this picture?", image("cat.png")])
await agent.run(["Summarize this report", file("report.pdf")])
await agent.run(["Which of these matches the invoice?", image("a.jpg"), image("b.jpg"), file("invoice.pdf")])
```

`image(...)` and `file(...)` take:

| Source                  | Example                                         | Sent as                              |
|-------------------------|-------------------------------------------------|--------------------------------------|
| a file path             | `image("cat.png")`, `file(Path("report.pdf"))`  | the bytes, inline (base64)           |
| bytes                   | `image(png_bytes, media_type="image/png")`      | the bytes, inline (base64)           |
| a web address           | `image("https://example.com/cat.png")`          | the address: the API downloads it    |

The media type of a path is guessed from its extension; bytes need `media_type=`. `file(...)` also takes a
`filename=` (by default the file's name), which the model sees.

!!! note "Files need their bytes"
    OpenAI downloads images from web addresses, but not files: give `file(...)` a path or bytes. A web
    address is only useful with OpenAI-compatible servers that download files themselves (like OpenRouter).

Which file types work depends on the model and the server. OpenAI models with vision read PNG, JPEG, WebP and
GIF images, and PDF files.

## In the history

`image(...)` and `file(...)` return plain dicts, and a user message with them has a list as its `content`:

```python
{
    "role": "user",
    "content": [
        {"type": "text", "text": "What is in this picture?"},
        {"type": "image", "url": "data:image/png;base64,iVBORw0KG..."},
        {"type": "file", "url": "data:application/pdf;base64,JVBERi0...", "filename": "report.pdf"},
    ],
}
```

They are kept in the history like any other message, so the model can see them on later turns. Inline images
and files are stored as base64 text, so they make stored conversations bigger: send a web address for images
when you can.

## Example

```python
--8<-- "examples/images_and_files.py"
```
