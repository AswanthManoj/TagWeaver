import re
import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString
from pydantic import BaseModel, Field
from typing import Any, ClassVar, Dict, List, Optional, Type, TypeVar, get_origin, get_args, Union, ForwardRef

T = TypeVar('T', bound='StructuredXML')

# =============================================================================
# StructuredXML Class for XML <-> Pydantic Model Handling
# =============================================================================

class StructuredXML(BaseModel):
    """
    Base class for Pydantic models that can generate XML schema templates,
    parse XML strings into model instances, and serialize instances to XML.

    Uses Pydantic field definitions for structure, types, and descriptions.
    XML tags directly correspond to Pydantic field names.
    """

    # Class variable to track forward references during schema generation
    _building_schema_for: ClassVar[set] = set()

    @classmethod
    def _get_model_root_tag(cls) -> str:
        """Gets the root tag name for the model (defaults to class name)."""
        return getattr(cls, '__xml_root_tag__', cls.__name__)

    @classmethod
    def _get_list_item_tag(cls, item_type: Type) -> str:
        """Determines the tag name for items within a list."""
        if isinstance(item_type, type) and issubclass(item_type, StructuredXML):
             return item_type._get_model_root_tag()
        else:
            return "item"

    # --- XML Schema Generation ---

    @classmethod
    def xml_schema(cls, indent_spaces: int = 4) -> str:
        """
        Generate an example XML structure template based on the Pydantic model.

        Args:
            indent_spaces: Number of spaces for indentation.

        Returns:
            A formatted XML string representing the model's structure.
        """
        root_tag = cls._get_model_root_tag()

        if root_tag in cls._building_schema_for:
            return f"<{root_tag}><!-- Recursive reference to {root_tag} --></{root_tag}>"

        cls._building_schema_for.add(root_tag)

        try:
            root = ET.Element(root_tag)
            cls._build_schema_element(root, cls)

            rough_string = ET.tostring(root, encoding='unicode')
            dom = parseString(rough_string)
            pretty_xml = dom.toprettyxml(indent=" " * indent_spaces)
            pretty_xml = '\n'.join(pretty_xml.split('\n')[1:]) # Remove declaration
            pretty_xml = pretty_xml.strip() + '\n'
            return pretty_xml

        finally:
            cls._building_schema_for.remove(root_tag)


    @classmethod
    def _build_schema_element(cls, parent_element: ET.Element, model_type: Type['StructuredXML']):
        """Recursively build the XML schema structure using ElementTree."""
        for field_name, field_info in model_type.model_fields.items():
            field_type = field_info.annotation
            description = field_info.description or f"Field '{field_name}'"
            comment_text = f" {description} "

            origin = get_origin(field_type)
            args = get_args(field_type)
            is_optional = False

            if origin is Union:
                union_args = [arg for arg in args if arg is not type(None)]
                if len(union_args) == 1 and type(None) in args:
                    field_type = union_args[0]
                    origin = get_origin(field_type)
                    args = get_args(field_type)
                    is_optional = True

            if isinstance(field_type, ForwardRef):
                 try:
                     global_ns = model_type.__module__.__dict__
                     local_ns = dict(vars(model_type))
                     field_type = field_type._evaluate(global_ns, local_ns, set())
                     origin = get_origin(field_type)
                     args = get_args(field_type)
                 except NameError:
                     el = ET.SubElement(parent_element, field_name)
                     el.append(ET.Comment(f" Type '{field_info.annotation}' (unresolved ForwardRef) - {description} "))
                     continue

            # Handle Lists
            if origin is list or origin is List:
                list_element = ET.SubElement(parent_element, field_name)
                list_element.append(ET.Comment(comment_text))
                item_type = args[0] if args else Any

                if item_type is Any:
                     item_element = ET.SubElement(list_element, "item")
                     item_element.append(ET.Comment(" Generic list item "))
                elif isinstance(item_type, type) and issubclass(item_type, StructuredXML):
                     item_tag = item_type._get_model_root_tag()
                     list_element.append(ET.Comment(f" Contains <{item_tag}> elements "))
                     item_element = ET.Element(item_tag)
                     cls._build_schema_element(item_element, item_type)
                     list_element.append(item_element)
                else:
                    item_tag = cls._get_list_item_tag(item_type)
                    item_element = ET.SubElement(list_element, item_tag)
                    item_element.append(ET.Comment(f" Example item of type {item_type.__name__} "))

            # Handle Nested StructuredXML Models
            elif isinstance(field_type, type) and issubclass(field_type, StructuredXML):
                nested_element = ET.SubElement(parent_element, field_name)
                nested_element.append(ET.Comment(comment_text))
                cls._build_schema_element(nested_element, field_type)

            # Handle Primitive Types
            else:
                primitive_element = ET.SubElement(parent_element, field_name)
                primitive_element.append(ET.Comment(comment_text))

    # --- XML Parsing ---

    @classmethod
    def parse_xml(cls: Type[T], xml_string: str) -> T:
        """
        Parses an XML string into an instance of this Pydantic model.
        Args:
            xml_string: The XML string to parse.
        Returns:
            An instance of the model populated with data from the XML.
        Raises:
            ValueError: If the XML is malformed or doesn't match the expected structure.
            pydantic.ValidationError: If the extracted data fails Pydantic validation.
        """
        try:
            cleaned_xml = re.sub(r'```xml\n?', '', xml_string, flags=re.IGNORECASE)
            cleaned_xml = cleaned_xml.replace('```', '').strip()
            if not cleaned_xml:
                raise ValueError("Input XML string is empty after cleaning")
            root_element = ET.fromstring(cleaned_xml)
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML format: {e}") from e
        except Exception as e:
             raise ValueError(f"Error preparing XML for parsing: {e}") from e

        expected_root_tag = cls._get_model_root_tag()
        if root_element.tag != expected_root_tag:
            raise ValueError(f"Expected root tag '<{expected_root_tag}>' but found '<{root_element.tag}>'")

        data = cls._parse_element_to_dict(root_element, cls)
        # Use Pydantic's model_validate to handle validation and instantiation
        # return cls(**data) # Old way
        return cls.model_validate(data) # Preferred Pydantic v2 way

    @classmethod
    def _parse_element_to_dict(cls, element: ET.Element, model_type: Type['StructuredXML']) -> Dict[str, Any]:
        """Recursively parses an XML element into a dictionary based on the model."""
        data: Dict[str, Any] = {}
        children_by_tag: Dict[str, List[ET.Element]] = {}
        for child in element:
            if isinstance(child.tag, str):
                 children_by_tag.setdefault(child.tag, []).append(child)

        for field_name, field_info in model_type.model_fields.items():
            field_type = field_info.annotation

            origin = get_origin(field_type)
            args = get_args(field_type)
            is_optional = False

            if origin is Union:
                union_args = [arg for arg in args if arg is not type(None)]
                if len(union_args) == 1 and type(None) in args:
                    field_type = union_args[0]
                    origin = get_origin(field_type)
                    args = get_args(field_type)
                    is_optional = True

            if isinstance(field_type, ForwardRef):
                 try:
                     global_ns = model_type.__module__.__dict__
                     local_ns = dict(vars(model_type)) # May need model_type's __dict__ depending on where Ref is defined
                     field_type = field_type._evaluate(global_ns, local_ns or None, set()) # Add recursive_guard
                     origin = get_origin(field_type)
                     args = get_args(field_type)
                 except NameError as e:
                      print(f"Warning: Could not resolve ForwardRef '{field_info.annotation}' for field '{field_name}'. Skipping field. Error: {e}")
                      continue

            child_elements = children_by_tag.get(field_name)
            if not child_elements:
                continue # Let Pydantic handle missing fields (default/required)

            # Handle Lists
            if origin is list or origin is List:
                item_type = args[0] if args else Any
                parsed_list = []
                list_container_element = child_elements[0] # Assume first element is the container

                if item_type is Any:
                     # Treat children as strings if type is Any
                     parsed_list = [child.text.strip() if child.text else None for child in list_container_element]
                elif isinstance(item_type, type) and issubclass(item_type, StructuredXML):
                    item_tag = item_type._get_model_root_tag()
                    list_item_elements = list_container_element.findall(item_tag)
                    for item_element in list_item_elements:
                         parsed_list.append(cls._parse_element_to_dict(item_element, item_type))
                else: # Primitive list items
                    item_tag = cls._get_list_item_tag(item_type)
                    list_item_elements = list_container_element.findall(item_tag)
                    for item_element in list_item_elements:
                        value_str = item_element.text.strip() if item_element.text else ""
                        # Parse non-empty strings or if the item itself might be optional (complex case)
                        # Let Pydantic handle validation of the parsed value (or None)
                        if value_str or item_type is type(None) or get_origin(item_type) is Union: # Attempt parse if non-empty or potentially optional type
                            try:
                                parsed_list.append(cls._parse_primitive(value_str, item_type))
                            except ValueError:
                                # If primitive parse fails, pass raw string to Pydantic
                                parsed_list.append(value_str)
                        else:
                             # Empty tag for a non-optional primitive type - Pydantic should catch this
                             # Pass the empty string for validation.
                              parsed_list.append(value_str)


                data[field_name] = parsed_list

            # Handle Nested StructuredXML Models (non-list)
            elif isinstance(field_type, type) and issubclass(field_type, StructuredXML):
                if len(child_elements) > 1:
                     print(f"Warning: Found multiple <{field_name}> elements for non-list field. Using the first one.")
                nested_element = child_elements[0]
                data[field_name] = cls._parse_element_to_dict(nested_element, field_type)

            # Handle Primitive Types (non-list)
            else:
                if len(child_elements) > 1:
                     print(f"Warning: Found multiple <{field_name}> elements for non-list primitive field. Using the first one.")
                primitive_element = child_elements[0]
                value_str = primitive_element.text.strip() if primitive_element.text else ""
                # Always pass the value string (even if empty) to Pydantic validation
                # Pydantic handles required/optional/default logic based on the raw value.
                try:
                    # Attempt primitive conversion, but pass raw string if it fails
                    data[field_name] = cls._parse_primitive(value_str, field_type)
                except ValueError:
                    data[field_name] = value_str # Let Pydantic validate the raw string

        return data

    @classmethod
    def _parse_primitive(cls, value: str, field_type: Type) -> Any:
        """Parses a string value into a primitive Python type. Returns raw string if specific type fails."""
        value = value.strip()
        # Handle target None type explicitly if needed
        if field_type is type(None):
             return None if not value else value # Return None if empty, else original string?

        # If value is empty string, return it directly for Pydantic to handle based on Optional/default
        if not value:
             return value

        # Attempt specific type conversions
        try:
            if field_type is str: return value
            if field_type is int: return int(value)
            if field_type is float: return float(value)
            if field_type is bool: return value.lower() in ('true', 'yes', '1', 'y')
            # Fallback for other types (enums, dates handled by Pydantic)
            return value
        except (ValueError, TypeError):
             # If direct conversion fails, return the original string
             # Pydantic's validation might still be able to parse it (e.g., for dates, enums)
             return value


    # --- XML Serialization ---

    def to_xml(self, pretty: bool = False, indent_spaces: int = 4) -> str:
        """
        Serializes this Pydantic model instance into an XML string.

        Args:
            pretty: If True, format the XML with indentation and newlines.
            indent_spaces: Number of spaces for indentation when pretty=True.

        Returns:
            An XML string representation of the model instance.
        """
        root_tag = self._get_model_root_tag()
        root = ET.Element(root_tag)
        model_data = self.model_dump(mode='python', by_alias=False)
        self._add_fields_to_element(root, model_data, self.__class__)

        rough_string = ET.tostring(root, encoding='unicode')

        if pretty:
            try:
                dom = parseString(rough_string)
                pretty_xml = dom.toprettyxml(indent=" " * indent_spaces)
                pretty_xml = '\n'.join(pretty_xml.split('\n')[1:])
                pretty_xml = pretty_xml.strip() + '\n'
                return pretty_xml
            except Exception as e:
                 print(f"Warning: Failed to pretty-print XML, returning raw string. Error: {e}")
                 return rough_string
        else:
            return rough_string

    def _add_fields_to_element(self, parent_element: ET.Element, data: Dict[str, Any], model_type: Type['StructuredXML']):
        """Recursively adds fields from data dict to XML element based on model type."""
        for field_name, field_value in data.items():
            if field_value is None:
                continue # Skip None values

            field_info = model_type.model_fields.get(field_name)
            if not field_info: continue

            field_type = field_info.annotation
            origin = get_origin(field_type)
            args = get_args(field_type)

            if origin is Union:
                union_args = [arg for arg in args if arg is not type(None)]
                if len(union_args) == 1 and type(None) in args:
                    field_type = union_args[0]
                    origin = get_origin(field_type)
                    args = get_args(field_type)

            if isinstance(field_type, ForwardRef):
                 try:
                     global_ns = model_type.__module__.__dict__
                     local_ns = dict(vars(model_type))
                     field_type = field_type._evaluate(global_ns, local_ns or None, set())
                     origin = get_origin(field_type)
                     args = get_args(field_type)
                 except NameError:
                      print(f"Warning: Could not resolve ForwardRef for serialization: {field_info.annotation}. Skipping field '{field_name}'.")
                      continue

            # Handle Lists
            if isinstance(field_value, list):
                list_element = ET.SubElement(parent_element, field_name)
                item_type = args[0] if args else Any
                for item in field_value:
                    if item is None: continue

                    item_model_type = None
                    # Determine the actual type of the item for tag generation/recursion
                    if isinstance(item, StructuredXML):
                         item_model_type = item.__class__
                    elif isinstance(item_type, type) and issubclass(item_type, StructuredXML):
                         item_model_type = item_type # Use the declared list item type

                    if item_model_type:
                         # Item is a nested model (either directly or via type hint)
                         item_tag = item_model_type._get_model_root_tag()
                         item_element = ET.SubElement(list_element, item_tag)
                         # Get item data (handle if it's already a model or still a dict)
                         item_data = item.model_dump(mode='python', by_alias=False) if isinstance(item, StructuredXML) else item
                         self._add_fields_to_element(item_element, item_data, item_model_type)
                    else:
                         # Item is primitive
                         item_tag = self._get_list_item_tag(item_type) # Use hinted type for tag name
                         item_element = ET.SubElement(list_element, item_tag)
                         item_element.text = str(item)

            # Handle Nested StructuredXML Models (or dicts matching model type)
            elif isinstance(field_value, StructuredXML) or (isinstance(field_value, dict) and isinstance(field_type, type) and issubclass(field_type, StructuredXML)):
                 nested_model_type = field_value.__class__ if isinstance(field_value, StructuredXML) else field_type
                 nested_element = ET.SubElement(parent_element, field_name)
                 nested_data = field_value.model_dump(mode='python', by_alias=False) if isinstance(field_value, StructuredXML) else field_value
                 self._add_fields_to_element(nested_element, nested_data, nested_model_type)

            # Handle Primitive Types
            else:
                element = ET.SubElement(parent_element, field_name)
                element.text = str(field_value)


# =============================================================================
# Instruction Template Classes
# =============================================================================

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
        all_vars = {**(self.variables or {}), **extra_vars}
        section_order = self.sections_order or [
            "task", "context", "response_schema", "guidelines",
            "examples", "tone", "custom_sections", "note"
        ]

        sections = []
        for section_name in section_order:
            # Allow referencing custom sections by name in the order list
            if section_name in (self.custom_sections or {}):
                 rendered = self._render_custom_section(section_name, vars=all_vars)
            else:
                 rendered = self._render_section(section_name, vars=all_vars)

            if rendered:
                sections.append(rendered)

        prompt = "\n\n".join(sections)
        # Final substitution pass on the whole prompt (optional, handles variables in headers/titles)
        # return self._substitute_variables(prompt, all_vars)
        return prompt # Substitution is done per-section, avoid double-substituting


    def _render_section(self, section: str, vars: Dict[str, Any]) -> Optional[str]:
        """Render a specific built-in section with variable substitution"""
        rendered_content = None

        if section == "task" and self.task:
            rendered_content = self._substitute_variables(self.task, vars)

        elif section == "context" and self.context:
            context = self._substitute_variables(self.context, vars)
            rendered_content = f"# Context:\n{context}"

        elif section == "response_schema" and self.response_schema:
            schema_str = ""
            # Check if it's a StructuredXML CLASS
            if isinstance(self.response_schema, type) and issubclass(self.response_schema, StructuredXML):
                # Use the xml_schema classmethod with correct argument name
                schema_str = self.response_schema.xml_schema(indent_spaces=2)
            # Check if it's a StructuredXML INSTANCE (less common for schema)
            elif isinstance(self.response_schema, StructuredXML):
                # Use the xml_schema classmethod via the instance's class
                schema_str = self.response_schema.__class__.xml_schema(indent_spaces=2)
            # Fallback to string representation
            else:
                schema_str = self._substitute_variables(str(self.response_schema), vars)

            # Format based on whether it looks like XML or not
            if schema_str.strip().startswith('<') and schema_str.strip().endswith('>'):
                 rendered_content = f"# Generate your response in the following XML format:\n\n```xml\n{schema_str}```"
            else:
                 rendered_content = f"# Response Format:\n{schema_str}"


        elif section == "guidelines" and self.guidelines:
            result = "# Guidelines:"
            for i, guideline in enumerate(self.guidelines, 1):
                if isinstance(guideline, str):
                    text = self._substitute_variables(guideline, vars)
                    result += f"\n{i}. {text}"
                elif isinstance(guideline, SubList):
                    header = self._substitute_variables(guideline.header, vars)
                    result += f"\n{i}. {header}:" # Added colon for clarity
                    for item in guideline.items:
                        item_text = self._substitute_variables(item, vars)
                        result += f"\n   - {item_text}"
            rendered_content = result

        elif section == "examples" and self.examples:
            result = "# Examples:" # Changed header slightly
            for i, example in enumerate(self.examples, 1):
                result += f"\n\n## Example {i}"
                input_text = self._substitute_variables(example.input, vars)
                output_text = self._substitute_variables(example.output, vars)
                result += f"\nInput:\n```\n{input_text}\n```" # Wrap in code blocks maybe?
                result += f"\nOutput:\n```xml\n{output_text}\n```" # Assuming XML output for examples
                if example.explanation:
                    explanation = self._substitute_variables(example.explanation, vars)
                    result += f"\n\n> Explanation: {explanation}" # Simpler note format
            rendered_content = result

        elif section == "tone" and self.tone:
            tone = self._substitute_variables(self.tone, vars)
            rendered_content = f"# Tone:\n{tone}"

        elif section == "note" and self.note:
            note = self._substitute_variables(self.note, vars)
            rendered_content = f"# Note:\n{note}" # Consistent header casing

        # This handles the "custom_sections" key if it's listed explicitly
        # in sections_order, rendering all custom sections together.
        elif section == "custom_sections" and self.custom_sections:
            custom_parts = []
            # Render only those custom sections NOT already handled individually
            handled_custom = [s for s in (self.sections_order or []) if s in self.custom_sections]
            for title, content in self.custom_sections.items():
                 if title not in handled_custom:
                    rendered_content = self._render_custom_section(title, content, vars)
                    if rendered_content:
                         custom_parts.append(rendered_content)
            rendered_content = "\n\n".join(custom_parts) if custom_parts else None

        return rendered_content


    def _render_custom_section(self, title: str, content: Optional[str] = None, vars: Optional[Dict[str, Any]] = None) -> Optional[str]:
         """Renders a specific custom section."""
         vars = vars or {}
         # Fetch content if not provided directly (e.g., called from render loop)
         content = content or (self.custom_sections or {}).get(title)
         if not content:
             return None
         # Format title nicely (e.g., 'my_section' -> '# My Section:')
         formatted_title = title.replace('_', ' ').title()
         content = self._substitute_variables(content, vars)
         return f"# {formatted_title}:\n{content}"


    def _substitute_variables(self, text: str, vars: Dict[str, Any]) -> str:
        """Replace {{variable}} patterns with their values"""
        if not text:
            return ""
        for var_name, value in vars.items():
            # Allow whitespace around variable name: {{ var }}
            pattern = r'\{\{\s*' + re.escape(str(var_name)) + r'\s*\}\}'
            # Convert value to string for substitution
            text = re.sub(pattern, str(value), text)
        return text
