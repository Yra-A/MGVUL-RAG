from common import constant


class VD_Prompt:

    default_sys_prompt = "You are a helpful assistant."

    @staticmethod
    def generate_detect_vul_prompt(code_snippet, vulnerability_knowledge) -> str:
        return f"""Task: I want you to act as a vulnerability detection expert. Given a target code snippet, determine whether it contains a defect based on the provided vulnerability knowledge. You will reason through the problem step by step and provide a conclusion.
Instructions:
1. Vulnerability Knowledge:
'''
{vulnerability_knowledge}
'''
2. Given Target Code:
'''
{code_snippet}
'''
3. Chain of Thought (COT) Reasoning:
Step 1: Please describe the intent of the given code.
Step 2: Check if the target code meets the preconditions_for_vulnerability and trigger_condition.
Step 3: If the target code meets Step 2, think about how to fix it.
Step 4: Verify if the target code fixes the vulnerability.

4. Conclusion:
Provide the conclusion: "{constant.vul_positive}" or "{constant.vul_negative}" based on whether the code contains the vulnerability.
Explanation: Offer a detailed explanation of why or why not the code matches the described vulnerability, based on the analysis of each step.
"""