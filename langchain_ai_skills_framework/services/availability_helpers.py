"""Shared helpers for formatting skill/script/resource availability messages.

These are used by multiple service classes to produce consistent
"not found – here's what's available" messages.
"""

from __future__ import annotations


from langchain_ai_skills_framework.loaders.skill_loader_protocol import (
    SkillLoaderProtocol,
)


async def format_skill_availability(
    *,
    loader: SkillLoaderProtocol,
    normalized_name: str,
    user_id: str,
) -> str:
    """Return a message listing available skills when *normalized_name* is missing or not found."""
    if user_id:
        summaries = await loader.list_all_summaries(user_id=user_id, allowed_skills=set())
        available_names = sorted(s.name for s in summaries)
    else:
        available_names = sorted(summary.name for summary in loader.list_skill_summaries(allowed_skills=set()))
    available = ", ".join(available_names)

    if normalized_name:
        prefix = f"Skill '{normalized_name}' not found."
    else:
        prefix = "No skill name provided."

    return f"{prefix} Available skills: {available or 'None configured'}"


async def format_script_availability(
    *,
    loader: SkillLoaderProtocol,
    skill_name: str,
    script_name: str,
    user_id: str,
    plugin_name: str | None = None,
) -> str:
    """Return a message listing available scripts for a given skill."""
    if user_id:
        script_names = await loader.list_skill_script_names_for_user(
            user_id=user_id, plugin_name=plugin_name, skill_name=skill_name
        )
    else:
        script_names = loader.list_skill_script_names(skill_name=skill_name, plugin_name=plugin_name)
    available_scripts = ", ".join(script_names)
    return f"Script '{script_name}' not found in skill '{skill_name}'. Available scripts: {available_scripts or 'none'}"


async def format_resource_availability(
    *,
    loader: SkillLoaderProtocol,
    skill_name: str,
    resource_name: str,
    user_id: str,
    plugin_name: str | None = None,
) -> str:
    """Return a message listing available resources for a given skill.

    Falls back to listing available skills if the skill itself isn't found.
    """
    from langchain_ai_skills_framework.loaders.exceptions.skill_not_found_error import (
        SkillNotFoundError,
    )

    try:
        if user_id:
            await loader.get_skill_details_for_user(user_id=user_id, plugin_name=plugin_name, skill_name=skill_name)
        else:
            loader.get_skill_details(skill_name=skill_name, plugin_name=plugin_name)
    except SkillNotFoundError:
        return await format_skill_availability(loader=loader, normalized_name=skill_name, user_id=user_id)

    if user_id:
        resource_names = await loader.list_skill_resource_names_for_user(
            user_id=user_id, plugin_name=plugin_name, skill_name=skill_name
        )
    else:
        resource_names = loader.list_skill_resource_names(skill_name=skill_name, plugin_name=plugin_name)
    available_resources = ", ".join(resource_names)
    return (
        f"Resource '{resource_name}' not found in skill '{skill_name}'. "
        f"Available resources: {available_resources or 'none'}"
    )
