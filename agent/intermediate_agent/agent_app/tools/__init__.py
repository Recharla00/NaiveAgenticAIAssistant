"""Tools available to the intermediate agent."""

from .calculator import add_numbers, multiply_numbers


def get_tools(*, include_search=False, include_calculator=True, include_demo_write=False):
    tools = [add_numbers, multiply_numbers] if include_calculator else []
    if include_search:
        from ..retrieval import search_internal_knowledge_base
        tools.append(search_internal_knowledge_base)
    if include_demo_write:
        from .demo_write import save_demo_note
        tools.append(save_demo_note)
    return tools
