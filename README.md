# TagWeaver
TagWeaver is a minimalist lightweight library for creating structured XML prompts for language models. It simplifies the process of generating, parsing, and validating structured outputs from LLMs through robust templating and schema definition.

## ✨ Features
- **Structured XML Generation**: Define output schemas declaratively using Pydantic models
- **Template-Based Prompting**: Create reusable instruction templates with variable substitution
- **Flexible Formatting**: Support for examples, guidelines, sublists, and custom sections
- **Parse with Confidence**: Extract structured data reliably from LLM responses

## 🚀 Installation
```bash
pip install git+https://github.com/AswanthManoj/TagWeaver.git
```

## 📝 Quick Start
```python
from tagweaver import StructuredXML, Field, InstructionTemplate, SubList
from typing import List, Optional

# =============================================================================
# Example Usage
# =============================================================================

# --- Define Models using StructuredXML ---
class Contact(StructuredXML):
    email: str = Field(..., description="Primary email address")
    phone: Optional[str] = Field(None, description="Contact phone number")

class Skill(StructuredXML):
    name: str = Field(..., description="Name of the skill")
    proficiency: str = Field(..., description="Skill proficiency level (e.g., Beginner, Intermediate, Expert)")
    years_experience: int = Field(..., description="Years of experience with this skill")

class Person(StructuredXML):
    """Represents a person with contact info and skills."""
    # Define root tag explicitly if needed (optional)
    # __xml_root_tag__ = 'PersonRecord'

    name: str = Field(..., description="Person's full name")
    age: int = Field(..., description="Person's age in years")
    bio: Optional[str] = Field(None, description="Short biography")
    contact: Contact = Field(..., description="Contact information")
    skills: List[Skill] = Field(default_factory=list, description="List of skills")
```

```python
# --- StructuredXML Usage ---

print("--- StructuredXML: Schema Generation ---")
person_schema = Person.xml_schema(indent_spaces=2)
print(person_schema)

print("\n--- StructuredXML: Parsing ---")
xml_input = """
<Person>
    <name>Alice Wonderland</name>
    <age>30</age>
    <bio>Curiouser and curiouser.</bio>
    <contact>
        <email>alice@example.com</email>
        <phone>123-456-7890</phone>
    </contact>
    <skills>
        <Skill>
            <name>Python</name>
            <proficiency>Expert</proficiency>
            <years_experience>5</years_experience>
        </Skill>
        <Skill>
            <name>XML Handling</name>
            <proficiency>Intermediate</proficiency>
            <years_experience>3</years_experience>
        </Skill>
    </skills>
</Person>
"""
try:
    parsed_person = Person.parse_xml(xml_input)
    print("Parsed Object:")
    print(parsed_person.model_dump_json(indent=2))

    print("\n--- StructuredXML: Serialization ---")
    xml_output = parsed_person.to_xml(pretty=True, indent_spaces=2)
    print("Serialized XML:")
    print(xml_output)

except (ValueError, Exception) as e:
    print(f"Error during StructuredXML processing: {e}")


# --- InstructionTemplate Usage ---

print("\n" + "="*30 + "\n")
print("--- InstructionTemplate Usage ---")

# 1. Create an InstructionTemplate instance
prompt_template = InstructionTemplate(
    task="Extract information about a person from the provided text and structure it as XML.",
    context="The following text contains details about a job applicant.",
    response_schema=Person, # Pass the StructuredXML class!
    guidelines=[
        "Extract the full name accurately.",
        "Ensure the age is a number.",
        SubList(header="Skill Proficiency Levels", items=["Use 'Beginner', 'Intermediate', or 'Expert' only."]),
        "If information is missing for an optional field (bio, phone), omit the tag.",
    ],
    examples=[
        Example(
            input="John Doe is 42. He knows Java (Expert) and lives at john@mail.com.",
            output="""<Person>
  <name>John Doe</name>
  <age>42</age>
  <contact>
    <email>john@mail.com</email>
  </contact>
  <skills>
    <Skill>
      <name>Java</name>
      <proficiency>Expert</proficiency>
      <years_experience>0</years_experience> <!-- Assume 0 if not mentioned -->
    </Skill>
  </skills>
</Person>""",
            explanation="Years of experience defaulted to 0 as it wasn't specified."
        )
    ],
    variables={"default_proficiency": "Intermediate"},
    custom_sections={"urgency": "Process this request with high priority."},
    # sections_order=["task", "context", "urgency", "response_schema", "guidelines", "examples"] # Optional custom order
)

# 2. Render the template
rendered_prompt = prompt_template.render()
print("Rendered Prompt:")
print(rendered_prompt)

print("\n--- Rendering with extra variables ---")
rendered_prompt_extra = prompt_template.render(applicant_text="Provided text snippet about the applicant.")
# Note: The example doesn't currently *use* applicant_text, but shows how to pass it.
# You would typically add a placeholder like {{ applicant_text }} in the 'task' or 'context'.
print(rendered_prompt_extra)
```

## 🧩 Core Components
- **StructuredXML**: Base class for defining structured output schemas
- **InstructionTemplate**: Configurable template for generating LLM instructions
- **Field**: Enhanced Pydantic field with XML-specific options
- **SubList**: Nested list structure for hierarchical guidelines

*TagWeaver: Weaving structure into AI responses without restricting grammer*
