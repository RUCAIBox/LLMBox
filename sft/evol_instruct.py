import json
import random
import wizardlm_utils as wizardlm_utils
import argparse


def main():
    parser = argparse.ArgumentParser(description="Generate instruction-following data.")

    parser.add_argument('--output_dir', default='alpaca_data_evol.json', help='Output directory for generated data')
    parser.add_argument(
        '--seed_tasks_path',
        default='alpaca_data_cleaned.json',
        help='Path to seed tasks file(https://github.com/gururise/AlpacaDataCleaned/blob/main/alpaca_data_cleaned.json)'
    )
    parser.add_argument(
        '--num_instructions_to_generate', type=int, default=100, help='Number of instructions to generate'
    )

    args = parser.parse_args()

    fr = open(args.seed_tasks_path, 'r')
    all_objs = json.load(fr)
    evol_objs = []
    num_success_ins = 0

    for cur_obj in all_objs:
        seed_instruction = cur_obj['instruction']
        instruction = cur_obj['instruction'].strip() + '\r\n' + cur_obj['input'].strip()

        # randomly choose one evol function generate the result
        evol_functions = [
            wizardlm_utils.createConstraintsPrompt, wizardlm_utils.createDeepenPrompt,
            wizardlm_utils.createConcretizingPrompt, wizardlm_utils.createReasoningPrompt,
            wizardlm_utils.createBreadthPrompt
        ]
        selected_function = random.choice(evol_functions)
        selected_evol_prompt = selected_function(instruction)

        # generate new instructions
        evol_instruction = wizardlm_utils.call_chatgpt(selected_evol_prompt)
        answer = wizardlm_utils.call_chatgpt(evol_instruction)

        # elimination part
        if wizardlm_utils.evol_elimination(seed_instruction, evol_instruction, answer):
            evol_objs.append({"instruction": evol_instruction, "output": answer})
            num_success_ins += 1

            print("seed:", seed_instruction)
            print("instruction:", evol_instruction, "output:", answer)
            print('evol success!')
            if (num_success_ins == args.num_instructions_to_generate):
                break

    with open(args.output_dir, 'w') as f:
        json.dump(evol_objs, f, indent=4)


if __name__ == "__main__":
    main()
