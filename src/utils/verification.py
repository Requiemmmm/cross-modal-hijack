"""
Verification utilities for attack results.
"""

import json
from typing import Dict, List, Optional
from collections import Counter


def analyze_token_distribution(image_tokens: List[int]) -> Dict:
    """
    Analyze distribution of generated image tokens.

    Args:
        image_tokens: List of generated image token IDs

    Returns:
        Dict with distribution statistics
    """
    if not image_tokens:
        return {
            'total_tokens': 0,
            'unique_tokens': 0,
            'mode_collapse': True,
        }

    counter = Counter(image_tokens)
    total = len(image_tokens)
    unique = len(counter)

    # Find most common token
    most_common = counter.most_common(1)[0]
    most_common_ratio = most_common[1] / total

    # Detect mode collapse (when one token dominates)
    mode_collapse = most_common_ratio > 0.3

    return {
        'total_tokens': total,
        'unique_tokens': unique,
        'most_common_token': most_common[0],
        'most_common_count': most_common[1],
        'most_common_ratio': most_common_ratio,
        'mode_collapse': mode_collapse,
        'top_5_tokens': counter.most_common(5),
    }


def check_token_validity(
    image_tokens: List[int],
    valid_range: tuple = (4, 8195),
) -> Dict:
    """
    Check if all image tokens are in valid range.

    Args:
        image_tokens: List of image token IDs
        valid_range: Tuple of (min_id, max_id) for valid image tokens

    Returns:
        Dict with validity information
    """
    min_id, max_id = valid_range
    invalid_tokens = [t for t in image_tokens if t < min_id or t > max_id]

    return {
        'total_tokens': len(image_tokens),
        'valid_tokens': len(image_tokens) - len(invalid_tokens),
        'invalid_tokens': len(invalid_tokens),
        'invalid_token_ids': invalid_tokens[:10],  # First 10 invalid
        'all_valid': len(invalid_tokens) == 0,
    }


def verify_attack_results(
    results: Dict,
    expected_text_contains: Optional[List[str]] = None,
    min_image_tokens: int = 100,
) -> Dict:
    """
    Verify attack results and provide summary.

    Args:
        results: Results dict from attack verification
        expected_text_contains: List of strings that should appear in generated text
        min_image_tokens: Minimum expected image tokens

    Returns:
        Dict with verification summary
    """
    summary = {
        'success': True,
        'issues': [],
    }

    # Check generated text
    generated_text = results.get('generated_text', '')
    if expected_text_contains:
        for expected in expected_text_contains:
            if expected.lower() not in generated_text.lower():
                summary['issues'].append(f"Expected '{expected}' in generated text")

    # Check image tokens
    image_tokens = results.get('image_tokens', [])
    if len(image_tokens) < min_image_tokens:
        summary['issues'].append(
            f"Too few image tokens: {len(image_tokens)} < {min_image_tokens}"
        )
        summary['success'] = False

    # Analyze token distribution
    distribution = analyze_token_distribution(image_tokens)
    summary['token_distribution'] = distribution

    if distribution['mode_collapse']:
        summary['issues'].append(
            f"Mode collapse detected: token {distribution['most_common_token']} "
            f"appears {distribution['most_common_ratio']:.1%} of time"
        )

    # Check token validity
    validity = check_token_validity(image_tokens)
    summary['token_validity'] = validity

    if not validity['all_valid']:
        summary['issues'].append(
            f"Found {validity['invalid_tokens']} invalid tokens"
        )

    # Overall success
    if summary['issues']:
        summary['success'] = len([i for i in summary['issues']
                                  if 'Too few' in i or 'invalid' in i]) == 0

    return summary


def load_and_verify(results_path: str, **kwargs) -> Dict:
    """
    Load results from file and verify.

    Args:
        results_path: Path to verification_results.json
        **kwargs: Additional arguments for verify_attack_results

    Returns:
        Dict with verification summary
    """
    with open(results_path, 'r') as f:
        results = json.load(f)

    return verify_attack_results(results, **kwargs)
