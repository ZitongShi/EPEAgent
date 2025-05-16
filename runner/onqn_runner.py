from typing import List, Dict, Any
import os
import json
from fold1.io_utils import ensure_dir, get_timestamp, dump_json
from fold1.profile_utils import filter_user_profile_by_label
from agents.agents import agent_1_market_data, agent_2_risk_assessment, agent_3_transaction_execution, \
    agent_4_diagnosis, agent_5_treatment_recommendation, agent_6_medication_management, epe_agent


__all__ = ['run_utility_test_ONQN', 'run_privacy_test_ONQN']


def _init_log(domain: str, suffix: str, output_dir: str) -> str:
    ensure_dir(output_dir)
    path = os.path.join(output_dir, f"{domain}_{suffix}_{get_timestamp()}.json")
    dump_json(path, [])
    return path


def run_utility_test_ONQN(args, user_profiles: List[Dict[str, Any]], questions: List[Dict[str, Any]],
                          domain: str = 'financial', epe_enabled: bool = False,
                          output_logdir: str = './GPT_3.5_turbo_log'):
    log_path = _init_log(domain, 'ONQN_utility', output_logdir)
    _run_onqn(args, user_profiles, questions, domain, epe_enabled, log_path, privacy_mode=False)


def run_privacy_test_ONQN(args, user_profiles: List[Dict[str, Any]], questions: List[Dict[str, Any]],
                           domain: str = 'financial', epe_enabled: bool = True,
                           output_logdir: str = './GPT_3.5_turbo_log'):
    log_path = _init_log(domain, 'ONQN_privacy', output_logdir)
    _run_onqn(args, user_profiles, questions, domain, epe_enabled, log_path, privacy_mode=True)


# ------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------

def _write_result(log_path: str, result: Dict[str, Any]):
    with open(log_path, 'r+', encoding='utf-8') as f:
        data = json.load(f)
        data.append(result)
        f.seek(0)
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.truncate()


def _run_onqn(args, user_profiles, questions, domain, epe_enabled, log_path, privacy_mode: bool):
    for uid, profile in enumerate(user_profiles):
        uname = profile.get('name', {}).get('value', f'User_{uid+1}')
        for q in questions:
            field, q_text = q.get('field', ''), q.get('question', '')
            # route based on domain
            if domain == 'financial':
                final_answer, inter = _financial_flow(args, profile, q, epe_enabled, privacy_mode)
            else:
                final_answer, inter = _medical_flow(args, profile, q, epe_enabled, privacy_mode)

            _write_result(log_path, {
                'user_index': uid,
                'user_name': uname,
                'domain': domain,
                'field': field,
                'question': q_text,
                'final_answer': final_answer,
                'intermediary_responses': inter,
                'epe_mode': epe_enabled,
                'privacy_mode': privacy_mode
            })
            print(f"[ONQN] ({'privacy' if privacy_mode else 'utility'}) {uname} => {field} DONE.")


# Domain‑specific flows ------------------------------------------------------

def _financial_flow(args, profile, q, epe_enabled, privacy_mode):
    inter = []
    if privacy_mode:
        labels = q.get('label', [])
        # select agents based on labels (1,2) or aggregator(3)
    user_profile_a1 = filter_user_profile_by_label(profile, [1], epe_enabled)
    user_profile_a2 = filter_user_profile_by_label(profile, [2], epe_enabled)
    r1 = agent_1_market_data(args, q, user_profile_a1, args.question_type, epe_enabled)
    r2 = agent_2_risk_assessment(args, q, user_profile_a2, args.question_type, epe_enabled)
    if r1: inter.append(r1)
    if r2: inter.append(r2)
    sanitized = epe_agent(args, inter, profile, 'financial', 3) if epe_enabled else inter
    user_profile_a3 = filter_user_profile_by_label(profile, [3], epe_enabled)
    final = agent_3_transaction_execution(args, sanitized, q, user_profile_a3, args.question_type, epe_enabled)
    return final, inter


def _medical_flow(args, profile, q, epe_enabled, privacy_mode):
    inter = []
    user_profile_a4 = filter_user_profile_by_label(profile, [4], epe_enabled)
    user_profile_a5 = filter_user_profile_by_label(profile, [5], epe_enabled)

    r4 = agent_4_diagnosis(args, q, user_profile_a4, args.question_type, epe_enabled)
    r5 = agent_5_treatment_recommendation(args, q, user_profile_a5, args.question_type, epe_enabled)
    if r4: inter.append(r4)
    if r5: inter.append(r5)
    sanitized = epe_agent(args, inter, profile, 'medical', 6) if epe_enabled else inter
    user_profile_a6 = filter_user_profile_by_label(profile, [6], epe_enabled)
    final = agent_6_medication_management(args, sanitized, q, user_profile_a6, args.question_type, epe_enabled)
    return final, inter
