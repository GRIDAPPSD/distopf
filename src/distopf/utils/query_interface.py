import re
from dataclasses import dataclass
from typing import List, Optional, Union
from enum import Enum


class QueryType(Enum):
    VOLTAGE = "voltage"
    CURRENT = "current"
    POWER_FLOW = "power_flow"
    ACTIVE_POWER = "active_power"
    REACTIVE_POWER = "reactive_power"
    GENERATION = "generation"
    LOAD = "load"
    CAPACITANCE = "capacitance"
    IMPEDANCE = "impedance"
    RESISTANCE = "resistance"


class Scope(Enum):
    ALL = "all"
    SPECIFIC = "specific"
    MULTIPLE = "multiple"


@dataclass
class ParsedQuery:
    query_type: QueryType
    scope: Scope
    elements: List[Union[int, str]]
    element_type: str


def create_word_to_number_mapping(max_number=1000):
    """Create automated word to number mapping up to max_number"""
    word_to_number = {}

    ones = [
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
    ]

    tens = [
        "",
        "",
        "twenty",
        "thirty",
        "forty",
        "fifty",
        "sixty",
        "seventy",
        "eighty",
        "ninety",
    ]

    for i, word in enumerate(ones):
        word_to_number[word] = i

    for ten_idx in range(2, 10):
        ten_word = tens[ten_idx]
        ten_value = ten_idx * 10
        word_to_number[ten_word] = ten_value

        for one_idx in range(1, 10):
            if ten_value + one_idx <= 99:
                word_to_number[f"{ten_word} {ones[one_idx]}"] = ten_value + one_idx
                word_to_number[f"{ten_word}-{ones[one_idx]}"] = ten_value + one_idx

    for i in range(1, min(10, max_number // 100 + 1)):
        word_to_number[f"{ones[i]} hundred"] = i * 100

    return word_to_number


class PowerSystemQueryParser:
    def __init__(self, max_word_number=1000):
        # Fixed keyword priorities - put more specific first to avoid conflicts
        self.generation_keywords = [
            "generation",
            "generations",
            "gen",
            "generator",
            "generators",
            "generate",
            "generating",
            "output",
        ]
        self.active_power_keywords = [
            "active power",
            "real power",
            "mw",
            "kw",
        ]  # Removed 'w' to avoid conflicts
        self.reactive_power_keywords = ["reactive power", "mvar", "kvar", "var"]
        self.power_flow_keywords = ["power flow", "flow", "flows", "power flows"]
        self.voltage_keywords = ["voltage", "volt", "volts", "voltages"]
        self.current_keywords = ["current", "currents", "amp", "amps", "ampere"]
        self.load_keywords = ["load", "loads", "demand", "demands", "consumption"]
        self.capacitance_keywords = [
            "capacitance",
            "capacitor",
            "capacitors",
            "cap",
            "farad",
            "farads",
        ]
        self.impedance_keywords = ["impedance"]
        self.resistance_keywords = ["resistance", "resistor"]

        self.element_keywords = {
            "bus": ["bus", "buses", "node", "nodes"],
            "branch": ["branch", "branches", "line", "lines", "transmission line"],
            "generator": ["generator", "gen", "generators"],
            "load": ["load", "loads"],
            "transformer": ["transformer", "transformers", "xfmr"],
            "substation": ["substation", "substations", "sub"],
        }

        self.word_to_number = create_word_to_number_mapping(max_word_number)
        self.ignore_words = {
            "at",
            "in",
            "of",
            "for",
            "the",
            "and",
            "or",
            "are",
            "is",
            "what",
            "show",
            "get",
            "optimal",
            "all",
            "every",
            "each",
            "total",
            "how",
            "value",
        }

    def extract_identifiers(self, text: str) -> List[Union[int, str]]:
        """Extract identifiers from quoted strings"""
        identifiers = []

        # Find all quoted strings (both single and double quotes)
        # This pattern handles both types of quotes properly
        all_quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", text)

        # Flatten the results (regex returns tuples)
        for single_quote, double_quote in all_quoted:
            if single_quote:
                identifiers.append(single_quote)
            elif double_quote:
                identifiers.append(double_quote)

        # Process identifiers
        processed_identifiers = []
        for identifier in identifiers:
            # Skip empty strings
            if not identifier.strip():
                continue

            if identifier.lower() in self.word_to_number:
                processed_identifiers.append(self.word_to_number[identifier.lower()])
            elif identifier.isdigit():
                processed_identifiers.append(int(identifier))
            else:
                processed_identifiers.append(identifier)

        return processed_identifiers

    def detect_query_type(self, text: str) -> Optional[QueryType]:
        """Detect query type with proper priority and no conflicts"""
        text_lower = text.lower()

        # Check in order of specificity to avoid conflicts
        # Put more specific keywords first
        if any(keyword in text_lower for keyword in self.generation_keywords):
            return QueryType.GENERATION
        elif any(keyword in text_lower for keyword in self.active_power_keywords):
            return QueryType.ACTIVE_POWER
        elif any(keyword in text_lower for keyword in self.reactive_power_keywords):
            return QueryType.REACTIVE_POWER
        elif any(keyword in text_lower for keyword in self.power_flow_keywords):
            return QueryType.POWER_FLOW
        elif any(keyword in text_lower for keyword in self.capacitance_keywords):
            return QueryType.CAPACITANCE
        elif any(keyword in text_lower for keyword in self.impedance_keywords):
            return QueryType.IMPEDANCE
        elif any(keyword in text_lower for keyword in self.resistance_keywords):
            return QueryType.RESISTANCE
        elif any(keyword in text_lower for keyword in self.load_keywords):
            return QueryType.LOAD
        elif any(keyword in text_lower for keyword in self.voltage_keywords):
            return QueryType.VOLTAGE
        elif any(keyword in text_lower for keyword in self.current_keywords):
            return QueryType.CURRENT

        return None

    def detect_element_type(self, text: str) -> str:
        """Detect element type with proper priority"""
        text_lower = text.lower()

        # Check for element keywords in order of priority
        # More specific keywords first to avoid conflicts
        for element_type, keywords in self.element_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return element_type

        return "bus"  # Default to bus

    def detect_scope(self, text: str, identifiers: List[Union[int, str]]) -> Scope:
        text_lower = text.lower()
        all_keywords = ["all", "every", "each", "total"]

        if any(keyword in text_lower for keyword in all_keywords):
            return Scope.ALL

        if identifiers:
            return Scope.SPECIFIC if len(identifiers) == 1 else Scope.MULTIPLE

        return Scope.ALL

    def parse_query(self, question: str) -> Optional[ParsedQuery]:
        text = question.strip()

        query_type = self.detect_query_type(text)
        if not query_type:
            return None

        element_type = self.detect_element_type(text)
        identifiers = self.extract_identifiers(text)
        scope = self.detect_scope(text, identifiers)

        if scope == Scope.ALL:
            elements = ["all"]
        else:
            elements = identifiers

        return ParsedQuery(
            query_type=query_type,
            scope=scope,
            elements=elements,
            element_type=element_type,
        )


def main():
    parser = PowerSystemQueryParser()

    print("Power System Query Interface")
    print("Use quotes around bus/branch names:")
    print("Examples:")
    print("  - generation for bus '56x', 'Nathan23', and 'fox'")
    print('  - voltage of bus "ABC123" and "Main_Bus"')
    print("  - load at all buses")
    print("Type 'quit' to exit\n")

    while True:
        question = input("Enter your question: ").strip()

        if question.lower() in ["quit", "exit", "q"]:
            print("Goodbye! Have an Optimal Day.")
            break

        if not question:
            print("Please enter a question.\n")
            continue

        parsed = parser.parse_query(question)

        if parsed:
            print("Parsed successfully:")
            print(f"  Type: {parsed.query_type.value}")
            print(f"  Element: {parsed.element_type}")
            print(f"  Scope: {parsed.scope.value}")
            print(f"  Elements: {parsed.elements}")
        else:
            print(
                "Could not understand the question. Try asking about voltage, current, load, generation, etc."
            )

        print()


if __name__ == "__main__":
    main()
