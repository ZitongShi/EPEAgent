
from typing import List, Dict, Any
import os, json
from fold1.io_utils import ensure_dir, get_timestamp
from fold1.profile_utils import filter_user_profile_by_label
from agents.agents import agent_1_market_data, agent_2_risk_assessment, agent_3_transaction_execution, \
    agent_4_diagnosis, agent_5_treatment_recommendation, agent_6_medication_management, epe_agent

__all__ = ['run_utility_test_MECE', 'run_privacy_test_MECE']


def _init_log(domain: str, suffix: str, output_dir: str):
    ensure_dir(output_dir)
    path = os.path.join(output_dir, f"{domain}_{suffix}_{get_timestamp()}.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump([], f, ensure_ascii=False, indent=2)
    return path


def run_utility_test_MECE(args, user_profiles: List[Dict[str, Any]], questions: List[Dict[str, Any]],
                          domain='financial', epe_enabled=False, output_logdir='./GPT_3.5_turbo_log'):
    log_path = _init_log(domain, 'MECE_utility', output_logdir)
    _run_mece(args, user_profiles, questions, domain, epe_enabled, log_path, privacy_mode=False)


def run_privacy_test_MECE(args, user_profiles: List[Dict[str, Any]], questions: List[Dict[str, Any]],
                           domain='financial', epe_enabled=True, output_logdir='./GPT_3.5_turbo_log'):
    log_path = _init_log(domain, 'MECE_privacy', output_logdir)
    _run_mece(args, user_profiles, questions, domain, epe_enabled, log_path, privacy_mode=True)


def _write(log_path, result):
    with open(log_path, 'r+', encoding='utf-8') as f:
        data = json.load(f)
        data.append(result)
        f.seek(0)
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.truncate()


def _mece_flow_financial(args, profile, q, epe_enabled):
    inter = []
    up1 = filter_user_profile_by_label(profile, [1], epe_enabled)
    up2 = filter_user_profile_by_label(profile, [2], epe_enabled)
    r1 = agent_1_market_data(args, q, up1, 'MECE', epe_enabled)
    r2 = agent_2_risk_assessment(args, q, up2, 'MECE', epe_enabled)
    if r1: inter.append(r1)
    if r2: inter.append(r2)
    sanitized = epe_agent(args, inter, profile, 'financial', 3) if epe_enabled else inter
    up3 = filter_user_profile_by_label(profile, [3], epe_enabled)
    final = agent_3_transaction_execution(args, sanitized, q, up3, 'MECE', epe_enabled)
    return final, inter


def _mece_flow_medical(args, profile, q, epe_enabled):
    inter = []
    up4 = filter_user_profile_by_label(profile, [4], epe_enabled)
    up5 = filter_user_profile_by_label(profile, [5], epe_enabled)
    r4 = agent_4_diagnosis(args, q, up4, 'MECE', epe_enabled)
    r5 = agent_5_treatment_recommendation(args, q, up5, 'MECE', epe_enabled)
    if r4: inter.append(r4)
    if r5: inter.append(r5)
    sanitized = epe_agent(args, inter, profile, 'medical', 6) if epe_enabled else inter
    up6 = filter_user_profile_by_label(profile, [6], epe_enabled)
    final = agent_6_medication_management(args, sanitized, q, up6, 'MECE', epe_enabled)
    return final, inter


def _run_mece(args, user_profiles, questions, domain, epe_enabled, log_path, privacy_mode):
    for uid, profile in enumerate(user_profiles):
        uname = profile.get('name', {}).get('value', f'User_{uid+1}')
        for q in questions:
            field = q.get('field', '')
            if domain == 'financial':
                final, inter = _mece_flow_financial(args, profile, q, epe_enabled)
            else:
                final, inter = _mece_flow_medical(args, profile, q, epe_enabled)
            _write(log_path, {
                'user_index': uid,
                'user_name': uname,
                'domain': domain,
                'field': field,
                'final_answer': final,
                'intermediary_responses': inter,
                'epe_mode': epe_enabled,
                'privacy_mode': privacy_mode
            })
            print(f"[MECE] ({'privacy' if privacy_mode else 'utility'}) {uname} => {field} DONE.")
