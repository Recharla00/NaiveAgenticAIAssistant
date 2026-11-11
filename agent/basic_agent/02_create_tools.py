"""Step 2: create tools the standalone agent can use.

Run from the project root:
    python agent/basic_agent/02_create_tools.py

A tool is a Python function with a clear name, inputs, and result.
For this lesson we call the tools ourselves. In step 3, the model will
choose a tool. No Ollama connection or policy documents are needed here.
"""


# The definitions now live in a shared file so all lessons use the same tools.
# The registry maps names to functions, without calling them yet.
from standalone_tools import TOOLS, add_numbers


def main():
    # 1. Call a tool directly, like any other Python function.
    print("Direct tool call:")
    print("add_numbers(20, 5) =", add_numbers(20, 5))

    # 2. Represent an action with a tool name and named arguments.
    # We write this request by hand for now; later the model will produce it.
    tool_name = "multiply_numbers"
    arguments = {"a": 125, "b": 8}

    # 3. Look up the function and execute it.
    # **arguments expands the dictionary into a=125, b=8.
    tool = TOOLS[tool_name]
    result = tool(**arguments)

    print("\nTool call using the registry:")
    print("Tool:", tool_name)
    print("Arguments:", arguments)
    print("Result:", result)


if __name__ == "__main__":
    main()
