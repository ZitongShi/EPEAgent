from typing import Dict, Any, List


def filter_user_profile_by_label(user_profile: Dict[str, Any],
                                 allowed_labels: List[int],
                                 epe_enabled: bool) -> Dict[str, Any]:
    """Return a view on the profile filtered by privacy labels when EPE is enabled."""
    if not epe_enabled:
        return user_profile
    return {
        k: v for k, v in user_profile.items()
        if any(label in allowed_labels for label in v.get('label', []))
    }


def find_answer_letter(text: str):
    import re
    match = re.findall(r'\((a|b|c|d)\)', text, flags=re.I)
    return match[-1].upper() if match else None
