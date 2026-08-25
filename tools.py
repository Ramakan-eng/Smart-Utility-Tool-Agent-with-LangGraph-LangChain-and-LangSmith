"""LangChain tools for the Smart Utility Tool Agent.

Each tool is a small, single-purpose function decorated with @tool so the
LLM can discover it through its name, docstring, and typed arguments and
decide when to call it.
"""

# -------------------------------------------------------
# Imports required by the tools
# -------------------------------------------------------

from langchain_core.tools import tool
# Imports the @tool decorator used to convert Python functions into LangChain tools.

from datetime import datetime
# Imports datetime functionality.

from zoneinfo import ZoneInfo
# Allows us to work with specific time zones.

import re
# Imports regular expressions for sentence counting.

import json
# Imports JSON functionality for reading the country dataset.


# -------------------------------------------------------
# Tool 1 — Calculator
# -------------------------------------------------------

@tool
def calculator(operation: str, a: float, b: float) -> str:
    """
    Perform basic arithmetic operations:
    add, subtract, multiply, divide, or percentage.
    """
    # Defines a calculator tool that the LLM can call.

    operation = operation.lower().strip()
    # Makes the operation name consistent.

    if operation == "add":

        return str(a + b)
        # Adds a and b.

    elif operation == "subtract":

        return str(a - b)
        # Subtracts b from a.

    elif operation == "multiply":

        return str(a * b)
        # Multiplies a and b.

    elif operation == "divide":

        if b == 0:
            return "Error: Cannot divide by zero"
            # Prevents division by zero.

        return str(a / b)
        # Divides a by b.

    elif operation == "percentage":

        return str((a / 100) * b)
        # Calculates a percent of b.

    return "Unsupported operation."
    # Returns an error for unsupported operations.


# -------------------------------------------------------
# Tool 2 — Current UK Date and Time
# -------------------------------------------------------

@tool
def current_datetime() -> str:
    """
    Return the current date and time in the United Kingdom.
    """
    # Defines a tool that provides the current UK date and time.

    now = datetime.now(
        ZoneInfo("Europe/London")
    )
    # Gets the current date and time in the Europe/London timezone.

    return now.strftime(
        "%d %B %Y, %I:%M:%S %p"
    )
    # Returns the date and time in a readable format.


# -------------------------------------------------------
# Tool 3 — Unit Conversion
# -------------------------------------------------------

@tool
def convert_units(
    value: float,
    from_unit: str,
    to_unit: str
) -> str:
    """
    Convert common units such as kilometers/miles,
    meters/feet, kilograms/pounds, and Celsius/Fahrenheit.
    """
    # Defines a unit conversion tool.

    from_unit = from_unit.lower().strip()
    # Standardizes the source unit.

    to_unit = to_unit.lower().strip()
    # Standardizes the destination unit.

    if from_unit == "kilometers" and to_unit == "miles":

        result = value * 0.621371

    elif from_unit == "miles" and to_unit == "kilometers":

        result = value * 1.60934

    elif from_unit == "meters" and to_unit == "feet":

        result = value * 3.28084

    elif from_unit == "feet" and to_unit == "meters":

        result = value * 0.3048

    elif from_unit == "kilograms" and to_unit == "pounds":

        result = value * 2.20462

    elif from_unit == "pounds" and to_unit == "kilograms":

        result = value * 0.453592

    elif from_unit == "celsius" and to_unit == "fahrenheit":

        result = (value * 9 / 5) + 32

    elif from_unit == "fahrenheit" and to_unit == "celsius":

        result = (value - 32) * 5 / 9

    else:

        return (
            f"Conversion from {from_unit} "
            f"to {to_unit} is not supported."
        )

    return f"{value} {from_unit} = {result:.4f} {to_unit}"
    # Returns the conversion result.


# -------------------------------------------------------
# Tool 4 — Text Counter
# -------------------------------------------------------

@tool
def text_counter(text: str) -> str:
    """
    Count the number of words, sentences, and characters
    in a piece of text.
    """
    # Defines a text analysis tool.

    words = len(text.split())
    # Counts words.

    characters = len(text)
    # Counts all characters, including spaces.

    characters_without_spaces = len(
        text.replace(" ", "")
    )
    # Counts characters excluding spaces.

    sentences = (
        len(re.findall(r"[^.!?]*[.!?]", text))
        or (1 if text.strip() else 0)
    )
    # Counts sentences ending with punctuation.
    # Counts non-empty text without punctuation as one sentence.

    return (
        f"Words: {words}, "
        f"Sentences: {sentences}, "
        f"Characters: {characters}, "
        f"Characters without spaces: "
        f"{characters_without_spaces}"
    )
    # Returns all text statistics.


# -------------------------------------------------------
# Tool 5 — Country Dataset Lookup
# -------------------------------------------------------

def load_countries():
    """
    Load country information from the JSON dataset.
    """

    with open(
        "data/countries.json",
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)
        # Loads and returns the JSON data.


@tool
def lookup_country_information(country: str) -> str:
    """
    Look up information about a country from the
    predefined country dataset.
    """
    # Defines a tool that searches the country dataset.

    countries = load_countries()
    # Loads all country records.

    for item in countries:
        # Loops through each country.

        if item["name"].lower() == country.lower():
            # Checks whether the country matches.

            return (
                f"Country: {item['name']}\n"
                f"Capital: {item['capital']}\n"
                f"Currency: {item['currency']}\n"
                f"Population: "
                f"{item['population_millions']} million\n"
                f"Continent: {item['continent']}"
            )
            # Returns the matching country information.

    return f"No information found for {country.title()}."
    # Returns a message if no matching country exists.


# -------------------------------------------------------
# All Tools
# -------------------------------------------------------

tools = [
    calculator,
    current_datetime,
    convert_units,
    text_counter,
    lookup_country_information
]
# Creates a list of all tools available to the agent.