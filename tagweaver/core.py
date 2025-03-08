import re
from textwrap import indent
import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Any, ClassVar, Dict, List, Optional, Type, TypeVar, get_origin, get_args, Union


T = TypeVar('T', bound='StructuredXML')


class StructuredXML(BaseModel):
    """Base class for parsing XML content into Pydantic models."""
    
    # Class variable for custom field parsers
    _custom_parsers: ClassVar[Dict[str, callable]] = {}
    
    @classmethod
    def extract_xml(cls, xml_string: str, tag: str, multiple: bool = False) -> Any:
        """Extract content between XML tags."""
        pattern = fr'(?s)<{tag}>(.*?)</{tag}>'
        
        if multiple:
            matches = re.finditer(pattern, xml_string)
            return [match.group(1).strip() for match in matches] if matches else []
        
        match = re.search(pattern, xml_string)
        return match.group(1).strip() if match else None
    
    @classmethod
    def clean_xml(cls, xml_string: str) -> str:
        """Remove markdown code blocks and other formatting."""
        if not xml_string:
            return ""
        cleaned = re.sub(r'```\w*\n?', '', xml_string)
        return cleaned.replace('```', '').replace('`', '')
    
    @classmethod
    def parse_xml(cls: Type[T], xml_string: str, clean_markdown: bool = False) -> T:
        """Parse XML string into model instance."""
        if clean_markdown: 
            xml_string = cls.clean_xml(xml_string)
        data = {}
        
        # Get field info from model
        for field_name, field_info in cls.model_fields.items():
            # Convert field_name from snake_case to kebab-case for XML tags
            xml_tag = field_name.replace('_', '-')
            field_type = field_info.annotation
            
            # Check for custom parser first
            if field_name in cls._custom_parsers:
                data[field_name] = cls._custom_parsers[field_name](xml_string, xml_tag)
                continue
            
            # Extract the raw value
            raw_value = cls.extract_xml(xml_string, xml_tag)
            if raw_value is None:
                continue
                
            # Handle different types
            try:
                data[field_name] = cls._parse_field_value(raw_value, field_type, xml_tag)
            except Exception as e:
                # If parsing fails, use default or None
                continue
        
        # Create the model instance
        return cls(**data)
    
    @classmethod
    def _parse_field_value(cls, value: str, field_type: Type, xml_tag: str) -> Any:
        """Parse a value according to its field type."""
        # Handle None
        if value is None:
            return None
            
        # Get origin type and args for generics
        origin = get_origin(field_type)
        args = get_args(field_type)
        
        # Handle Optional types
        if origin is Union and type(None) in args:
            inner_type = next(arg for arg in args if arg is not type(None))
            return cls._parse_field_value(value, inner_type, xml_tag)
        
        # Handle lists
        if origin is list:
            item_type = args[0] if args else Any
            # Check if items are wrapped in their own tags
            if issubclass(item_type, StructuredXML):
                # For nested structured objects in a list
                items = []
                # Try to extract items with an item-specific tag
                item_tag = item_type.__name__.lower().replace('_', '-')
                item_texts = cls.extract_xml(value, item_tag, multiple=True)
                
                # If no items found with specific tag, try generic 'item' tag
                if not item_texts:
                    item_texts = cls.extract_xml(value, 'item', multiple=True)
                
                # If still no items, treat the whole value as one item
                if not item_texts:
                    item_texts = [value]
                
                for item_text in item_texts:
                    items.append(item_type.parse_xml(item_text))
                return items
            else:
                # For primitive types in a list
                # First check for li tags
                li_items = re.findall(r'<li>(.*?)</li>', value)
                if li_items:
                    return [cls._parse_primitive(item.strip(), item_type) for item in li_items]
                elif ',' in value:
                    # Assume comma-separated values
                    return [cls._parse_primitive(item.strip(), item_type) for item in value.split(',')]
                else:
                    # Might be newline-separated
                    return [cls._parse_primitive(item.strip(), item_type) for item in value.split('\n') if item.strip()]
        
        # Handle nested StructuredXML objects
        if isinstance(field_type, type) and issubclass(field_type, StructuredXML):
            return field_type.parse_xml(value)
        
        # Handle primitive types
        return cls._parse_primitive(value, field_type)
    
    @classmethod
    def _parse_primitive(cls, value: str, field_type: Type) -> Any:
        """Parse primitive values."""
        if field_type is str:
            return value
        elif field_type is int:
            return int(value)
        elif field_type is float:
            return float(value)
        elif field_type is bool:
            return value.lower() in ('true', 'yes', '1', 'y')
        else:
            # Fall back to string
            return value
    
    @classmethod
    def register_parser(cls, field_name: str, parser_func: callable):
        """Register a custom parser for a field."""
        cls._custom_parsers[field_name] = parser_func
    
    @classmethod
    def xml_schema(cls, indent_level: int = 0) -> str:
        """Generate XML schema representation of the model.
        
        This creates a template showing the expected XML structure with
        field descriptions from the Pydantic model's Field descriptions.
        """
        schema_lines = []
        indent_str = ' ' * 4
        
        for field_name, field_info in cls.model_fields.items():
            # Convert field_name from snake_case to kebab-case for XML tags
            xml_tag = field_name.replace('_', '-')
            field_type = field_info.annotation
            
            # Get field description if available
            field_description = field_info.description or f"The {field_name} field"
            
            # Handle different field types
            origin = get_origin(field_type)
            args = get_args(field_type)
            
            # Handle Optional types
            if origin is Union and type(None) in args:
                inner_type = next(arg for arg in args if arg is not type(None))
                field_type = inner_type
                origin = get_origin(inner_type)
                args = get_args(inner_type)
            
            # Handle lists
            if origin is list:
                item_type = args[0] if args else Any
                schema_lines.append(f"<{xml_tag}>")
                schema_lines.append(f"{indent_str}<!-- {field_description} -->")
                
                if issubclass(item_type, StructuredXML):
                    # For lists of structured objects
                    item_tag = item_type.__name__.lower().replace('_', '-')
                    item_schema = item_type.xml_schema(indent_level=1)
                    # Indent the nested schema
                    indented_schema = indent(item_schema, indent_str)
                    schema_lines.append(indented_schema)
                else:
                    # For lists of primitive types
                    schema_lines.append(f"{indent_str}<li><!-- List item 1 of type {item_type.__name__} --></li>")
                    schema_lines.append(f"{indent_str}<li><!-- List item 2 of type {item_type.__name__} --></li>")
                    schema_lines.append(f"{indent_str}...")
                
                schema_lines.append(f"</{xml_tag}>")
            
            # Handle nested StructuredXML objects
            elif isinstance(field_type, type) and issubclass(field_type, StructuredXML):
                schema_lines.append(f"<{xml_tag}>")
                schema_lines.append(f"{indent_str}<!-- {field_description} -->")
                
                # Get the nested schema
                nested_schema = field_type.xml_schema(indent_level=1)
                # Indent the nested schema
                indented_schema = indent(nested_schema, indent_str)
                schema_lines.append(indented_schema)
                
                schema_lines.append(f"</{xml_tag}>")
            
            # Handle primitive types
            else:
                schema_lines.append(f"<{xml_tag}><!-- {field_description} --></{xml_tag}>")
        
        # Return the schema with proper indentation
        tag_name = cls._get_tag_name(cls)
        return f"<{tag_name}>\n{indent_str}{f'\n{indent_str}'.join(schema_lines)}\n</{tag_name}>"
    
    @field_validator('*', mode='before')
    @classmethod
    def empty_str_to_none(cls, value: Any, info):
        """Convert empty strings to None."""
        if value == "":
            return None
        return value

    @classmethod
    def _get_tag_name(cls, class_type: Type) -> str:
        if hasattr(class_type, '__tagname__'):
            return class_type.__tagname__

        class_name = class_type.__name__
        
        result = ""
        prev_is_upper = False
        
        for i, char in enumerate(class_name):
            if char.isupper():
                if i > 0 and not prev_is_upper:
                    result += "-"
                result += char.lower()
                prev_is_upper = True
            else:
                result += char
                prev_is_upper = False
        return result.replace('_', '-')


class SubList(BaseModel):
    """A header with a nested list of items"""
    header: str
    items: List[str]

class Example(BaseModel):
    """Input/output examples for few-shot learning"""
    input: str
    output: str
    explanation: Optional[str] = None

class InstructionTemplate(BaseModel):
    """Template for generating structured instruction prompts for language models"""
    # Core components
    task: str = Field(..., description="Main instruction describing what the model should do")
    
    # Optional components
    context: Optional[str] = Field(None, description="Background information or scenario details")
    response_schema: Optional[Union[StructuredXML, Type[StructuredXML], str]] = Field(None, description="Expected output format (XML schema or description)")
    guidelines: Optional[List[Union[str, SubList]]] = Field(None, description="Rules or constraints to follow")
    examples: Optional[List[Example]] = Field(None, description="Few-shot learning examples")
    tone: Optional[str] = Field(None, description="Desired tone for the response")
    note: Optional[str] = Field(None, description="Additional notes or reminders")
    
    # Customization options
    custom_sections: Optional[Dict[str, str]] = Field(None, description="Additional custom sections")
    sections_order: Optional[List[str]] = Field(None, description="Custom ordering of sections")
    variables: Optional[Dict[str, Any]] = Field(None, description="Template variables for substitution")
    
    def render(self, **extra_vars) -> str:
        """Render the instruction template as a complete prompt string"""
        # Combine default and provided variables
        all_vars = {**(self.variables or {}), **extra_vars}
        
        # Default section order if not specified
        section_order = self.sections_order or [
            "task", "context", "response_schema", "guidelines", 
            "examples", "tone", "custom_sections", "note"
        ]
        
        sections = []
        for section in section_order:
            rendered = self._render_section(section, all_vars)
            if rendered:
                sections.append(rendered)
        
        # Join sections and apply final variable substitution
        prompt = "\n\n".join(sections)
        return self._substitute_variables(prompt, all_vars)
    
    def _render_section(self, section: str, vars: Dict[str, Any]) -> Optional[str]:
        """Render a specific section with variable substitution"""
        if section == "task" and self.task:
            return self._substitute_variables(self.task, vars)
            
        elif section == "context" and self.context:
            context = self._substitute_variables(self.context, vars)
            return f"# Context:\n{context}"
            
        elif section == "response_schema" and self.response_schema:
            if isinstance(self.response_schema, type) and issubclass(self.response_schema, StructuredXML):
                schema = self.response_schema.xml_schema(indent_level=2)
                return f"# Generate your response in the following XML format:\n\n```xml\n{schema}\n```"
            elif hasattr(self.response_schema, 'xml_schema'):
                schema = self.response_schema.xml_schema(indent_level=2)
                return f"# Generate your response in the following XML format:\n\n```xml\n{schema}\n```"
            else:
                schema = self._substitute_variables(str(self.response_schema), vars)
                return f"# Response Format:\n{schema}"
                
        elif section == "guidelines" and self.guidelines:
            result = "# Guidelines:"
            for i, guideline in enumerate(self.guidelines, 1):
                if isinstance(guideline, str):
                    text = self._substitute_variables(guideline, vars)
                    result += f"\n{i}. {text}"
                elif isinstance(guideline, SubList):
                    header = self._substitute_variables(guideline.header, vars)
                    result += f"\n{i}. {header}"
                    for item in guideline.items:
                        item_text = self._substitute_variables(item, vars)
                        result += f"\n   - {item_text}"
            return result
            
        elif section == "examples" and self.examples:
            result = "# Example Inputs and Outputs:"
            for i, example in enumerate(self.examples, 1):
                result += f"\n## Example {i}:"
                input_text = self._substitute_variables(example.input, vars)
                output_text = self._substitute_variables(example.output, vars)
                result += f"\nInput: {input_text}"
                result += f"\nOutput: {output_text}"
                if example.explanation:
                    explanation = self._substitute_variables(example.explanation, vars)
                    result += f"\n\n> Note: Explanation: {explanation}"
            return result
            
        elif section == "tone" and self.tone:
            tone = self._substitute_variables(self.tone, vars)
            return f"\n\n# Tone:\n{tone}"
            
        elif section == "note" and self.note:
            note = self._substitute_variables(self.note, vars)
            return f"\n# Note: \n{note}"
            
        elif section == "custom_sections" and self.custom_sections:
            custom_parts = []
            for title, content in self.custom_sections.items():
                content = self._substitute_variables(content, vars)
                custom_parts.append(f"{title}:\n{content}")
            return "\n\n".join(custom_parts)
            
        elif self.custom_sections and section in self.custom_sections:
            content = self._substitute_variables(self.custom_sections[section], vars)
            return f"{section.replace('_', ' ').title()}:\n{content}"
            
        return None
    
    def _substitute_variables(self, text: str, vars: Dict[str, Any]) -> str:
        """Replace {{variable}} patterns with their values"""
        if not text:
            return ""
        for var, value in vars.items():
            pattern = r'\{\{\s*' + re.escape(str(var)) + r'\s*\}\}'
            text = re.sub(pattern, str(value), text)
        return text
