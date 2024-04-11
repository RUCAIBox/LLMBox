from .sftdataset import SFTDataset


class UltraDataset(SFTDataset):
    """
    UltraChat is composed of three sectors:
    🌏 Questions about the World: The dialogue data in this sector is derived from a wide range of inquiries related to concepts, entities, and objects from the real world. The topics covered are extensive, spanning areas such as technology, art, and entrepreneurship.
    ✍🏻 Writing and Creation: The dialogue data in this sector is driven by the demands for writing/creation from scratch, and encompasses any tasks that an AI assistant may aid within the creative process, spanning from email composition to crafting narratives and plays, and beyond.
    📋 Assistance on Existent Materials: The dialogue data in this sector is generated based on existing materials, including but not limited to rewriting, continuation, summarization, and inference, covering a diverse range of topics.
    """

    instruction_template = "\n\n### Human:\n"
    response_template = "\n\n### Assistant:\n"

    # single round process
    def process(self, tokenizer):
            """Process the dataset for the first round of dialogue and return input_ids and labels."""
            input_ids = []
            labels = []
            list_data_dict = self.load_data()
            for example in list_data_dict:
                first_user_message = next((msg for msg in example['messages'] if msg['role'] == 'user'), None)
                first_assistant_message = next((msg for msg in example['messages'] if msg['role'] == 'assistant'), None)
                if first_user_message and first_assistant_message:
                    user_content = first_user_message['content'].strip()
                    assistant_content = first_assistant_message['content'].strip()
                    s = user_content
                    t = assistant_content

                    input_id, label = self.encode_src_tgt(s, t, tokenizer)
                    input_ids.append(input_id)
                    labels.append(label)

            return input_ids, labels