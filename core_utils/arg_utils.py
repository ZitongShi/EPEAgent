import argparse

def parse_args():
    """Centralised CLI argument parsing."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        choices=[
                            'gemini-1.5-flash-002', 'gpt-3.5-turbo',
                            'gemini-1.5-pro-001', 'gpt-4o',
                            'claude-3-5-haiku-20241022'
                        ],
                        default='claude-3-5-haiku-20241022')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--domain', type=str, choices=['financial', 'medical'], default='financial',
                        help='Decide whether to run financial or medical experiment.')
    parser.add_argument('--task', type=str, choices=['utility', 'privacy'], default='utility')
    parser.add_argument('--epe_enabled', action='store_true',
                        help='Whether to enable EPE-Agent pipeline (privacy enhancement).')
    parser.add_argument('--profile_path', type=str, default='./user_profiles/financial_profiles.json',
                        help='Path to the user profile JSON.')
    parser.add_argument('--questions_path', type=str, default='',
                        help='Path to the questions JSON. If empty, we use a built‑in example.')
    parser.add_argument('--output_logdir', type=str, default='./GPT_3.5_turbo_log',
                        help='Directory to save the output logs.')
    parser.add_argument('--question_type', type=str, choices=['MECE', 'ONQN'], default='MECE',
                        help='Type of question set to run.')
    return parser.parse_args()