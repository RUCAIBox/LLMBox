import json


def truncate_filter(input_path, save_path, truncate_length=2048):
    # Load JSON data
    with open(input_path, encoding="utf-8") as f:
        dataset = json.load(f)
    # Truncate strings and filter the 'conversations' key
    filtered_conversations = []
    for row in dataset:
        conversations = row.pop('conversations', [])
        valid_conversations = [{
            "from": str(conv["from"][:truncate_length]).replace('\u2028', '').replace('\u2029', ''),
            "value": str(conv["value"][:truncate_length]).replace('\u2028', '').replace('\u2029', '')
        } for conv in conversations if isinstance(conv.get("from"), str) and isinstance(conv.get("value"), str)]
        filtered_conversations.append({"conversations": valid_conversations})
    # Write the filtered and truncated 'conversations' key to a new JSON file
    with open(save_path, 'w', encoding='utf-8') as output_file:
        json.dump(filtered_conversations, output_file, ensure_ascii=False, indent=2)
