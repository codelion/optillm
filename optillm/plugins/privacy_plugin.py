import spacy
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine, DeanonymizeEngine, OperatorConfig
from presidio_anonymizer.operators import Operator, OperatorType

from typing import Dict, Tuple

SLUG = "privacy"

class InstanceCounterAnonymizer(Operator):
    """
    Anonymizer which replaces the entity value
    with an instance counter per entity.
    """

    REPLACING_FORMAT = "<{entity_type}_{index}>"

    def operate(self, text: str, params: Dict = None) -> str:
        """Anonymize the input text."""

        entity_type: str = params["entity_type"]

        # entity_mapping is a dict of dicts containing mappings per entity type
        entity_mapping: Dict[Dict:str] = params["entity_mapping"]

        entity_mapping_for_type = entity_mapping.get(entity_type)
        if not entity_mapping_for_type:
            new_text = self.REPLACING_FORMAT.format(
                entity_type=entity_type, index=0
            )
            entity_mapping[entity_type] = {}

        else:
            if text in entity_mapping_for_type:
                return entity_mapping_for_type[text]

            previous_index = self._get_last_index(entity_mapping_for_type)
            new_text = self.REPLACING_FORMAT.format(
                entity_type=entity_type, index=previous_index + 1
            )

        entity_mapping[entity_type][text] = new_text
        return new_text

    @staticmethod
    def _get_last_index(entity_mapping_for_type: Dict) -> int:
        """Get the last index for a given entity type."""

        def get_index(value: str) -> int:
            return int(value.split("_")[-1][:-1])

        indices = [get_index(v) for v in entity_mapping_for_type.values()]
        return max(indices)

    def validate(self, params: Dict = None) -> None:
        """Validate operator parameters."""

        if "entity_mapping" not in params:
            raise ValueError("An input Dict called `entity_mapping` is required.")
        if "entity_type" not in params:
            raise ValueError("An entity_type param is required.")

    def operator_name(self) -> str:
        return "entity_counter"

    def operator_type(self) -> OperatorType:
        return OperatorType.Anonymize

def download_model(model_name):
    if not spacy.util.is_package(model_name):
        print(f"Downloading {model_name} model...")
        spacy.cli.download(model_name)
    else:
        print(f"{model_name} model already downloaded.")

def replace_entities(entity_map, text):
    # Create a reverse mapping of placeholders to entity names
    reverse_map = {}
    for entity_type, entities in entity_map.items():
        for entity_name, placeholder in entities.items():
            reverse_map[placeholder] = entity_name
    
    # Function to replace placeholders with entity names
    def replace_placeholder(match):
        placeholder = match.group(0)
        return reverse_map.get(placeholder, placeholder)
    
    # Use regex to find and replace all placeholders
    import re
    pattern = r'<[A-Z_]+_\d+>'
    replaced_text = re.sub(pattern, replace_placeholder, text)
    
    return replaced_text

def run(system_prompt: str, initial_query: str, client, model: str) -> Tuple[str, int]:
    # Use the function
    model_name = "en_core_web_lg"
    download_model(model_name)

    analyzer = AnalyzerEngine() 
    analyzer_results = analyzer.analyze(text=initial_query, language="en")

    # Create Anonymizer engine and add the custom anonymizer
    anonymizer_engine = AnonymizerEngine()
    anonymizer_engine.add_anonymizer(InstanceCounterAnonymizer)

    # Create a mapping between entity types and counters
    entity_mapping = dict()

    # Anonymize the text
    anonymized_result = anonymizer_engine.anonymize(
        initial_query,
        analyzer_results,
        {
            "DEFAULT": OperatorConfig(
                "entity_counter", {"entity_mapping": entity_mapping}
            )
        },
    )
    # print(f"Anonymized request: {anonymized_result.text}")
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": anonymized_result.text}],
    )

    # print(entity_mapping)
    final_response = response.choices[0].message.content.strip()
    # print(f"response: {final_response}")

    final_response = replace_entities(entity_mapping, final_response)
    
    return final_response, response.usage.completion_tokens