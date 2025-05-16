
import os, json
from fold1.arg_utils import parse_args
from fold1.io_utils import load_json
from fold2.mece_runner import run_utility_test_MECE, run_privacy_test_MECE
from fold2.onqn_runner import run_utility_test_ONQN, run_privacy_test_ONQN


def _load_dataset(args):
    """Return user_profiles and questions according to CLI options."""
    domain = args.domain
    qtype = args.question_type
    task  = args.task

    profile_file = f'user_profiles/{domain}_profiles.json'
    if args.profile_path:
        profile_file = args.profile_path
    user_profiles = load_json(profile_file)

    if args.questions_path:
        question_file = args.questions_path
    else:
        prefix = 'Utility' if task == 'utility' else 'Privacy'
        qtype_folder = 'Multiple_choice_data' if qtype == 'MECE' else 'Open_questions_data'
        question_file = f'{prefix}_{qtype_folder}/{domain}_{qtype}.json'
    questions = load_json(question_file)
    return user_profiles, questions


def main():
    args = parse_args()
    os.environ.setdefault('OPENAI_API_KEY', '')  # ensure key exists
    user_profiles, questions = _load_dataset(args)

    if args.question_type == 'MECE':
        if args.task == 'utility':
            run_utility_test_MECE(args, user_profiles, questions, args.domain, args.epe_enabled, args.output_logdir)
        else:
            run_privacy_test_MECE(args, user_profiles, questions, args.domain, args.epe_enabled, args.output_logdir)
    else:  # ONQN
        if args.task == 'utility':
            run_utility_test_ONQN(args, user_profiles, questions, args.domain, args.epe_enabled, args.output_logdir)
        else:
            run_privacy_test_ONQN(args, user_profiles, questions, args.domain, args.epe_enabled, args.output_logdir)

    print('Done. Logs saved to', args.output_logdir)


if __name__ == '__main__':
    main()
