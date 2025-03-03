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

# Define your structured output schema
class ProductReview(StructuredXML):
    product_name: str = Field(..., description="Name of the reviewed product")
    rating: int = Field(..., description="Rating from 1-5 stars")
    pros: List[str] = Field(..., description="Positive aspects of the product")
    cons: List[str] = Field(..., description="Negative aspects of the product")
    summary: str = Field(..., description="Brief summary of the review")

# Create an instruction template
template = InstructionTemplate(
    task="Write a detailed product review based on the customer feedback.",
    context="You are analyzing customer feedback for an e-commerce platform.",
    response_schema=ProductReview,
    guidelines=[
        "Be objective and balanced in your assessment",
        SubList(
            header="Consider these aspects in your review:",
            items=[
                "Value for money",
                "Build quality",
                "User experience"
            ]
        )
    ]
)

# Generate the prompt
prompt = template.render()

# Later, parse the LLM response
review = ProductReview.parse_xml(llm_response)
print(f"Rating: {review.rating}/5 stars")
```

## 🧩 Core Components
- **StructuredXML**: Base class for defining structured output schemas
- **InstructionTemplate**: Configurable template for generating LLM instructions
- **Field**: Enhanced Pydantic field with XML-specific options
- **SubList**: Nested list structure for hierarchical guidelines

*TagWeaver: Weaving structure into AI responses without restricting grammer*
