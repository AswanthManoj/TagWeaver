import re
import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString
from pydantic import BaseModel, Field
from typing import Any, ClassVar, Dict, List, Optional, Type, TypeVar, get_origin, get_args, Union, ForwardRef

T = TypeVar('T', bound='StructuredXML')

# =============================================================================
# StructuredXML Class (with Case-Insensitive Parsing)
# =============================================================================

class StructuredXML(BaseModel):
    """
    Base class for Pydantic models that can generate XML schema templates,
    parse XML strings case-insensitively into model instances, and serialize
    instances back to XML using canonical casing.

    Uses Pydantic field definitions for structure, types, and descriptions.
    XML tags during parsing are matched case-insensitively to Pydantic field names.
    XML tags during serialization use the exact Pydantic field names.
    """
    _building_schema_for: ClassVar[set] = set() # For schema recursion check

    @classmethod
    def _get_model_root_tag(cls) -> str:
        """Gets the canonical root tag name for the model (defaults to class name)."""
        return getattr(cls, '__xml_root_tag__', cls.__name__)

    @classmethod
    def _get_list_item_tag(cls, item_type: Type) -> str:
        """Determines the canonical tag name for items within a list."""
        if isinstance(item_type, type) and issubclass(item_type, StructuredXML):
             return item_type._get_model_root_tag()
        else:
            return "item"

    # --- XML Schema Generation (Case-Sensitive Output) ---

    @classmethod
    def xml_schema(cls, indent_spaces: int = 4) -> str:
        # (Schema generation logic remains unchanged - generates canonical case)
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
            pretty_xml = '\n'.join(pretty_xml.split('\n')[1:])
            pretty_xml = pretty_xml.strip() + '\n'
            return pretty_xml
        finally:
            cls._building_schema_for.remove(root_tag)

    @classmethod
    def _build_schema_element(cls, parent_element: ET.Element, model_type: Type['StructuredXML']):
        # (Schema building logic remains unchanged - uses canonical case)
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
                    field_type = union_args[0]; origin = get_origin(field_type); args = get_args(field_type); is_optional = True
            if isinstance(field_type, ForwardRef):
                 try:
                     global_ns, local_ns = model_type.__module__.__dict__, dict(vars(model_type))
                     field_type = field_type._evaluate(global_ns, local_ns or None, set())
                     origin = get_origin(field_type); args = get_args(field_type)
                 except NameError:
                     ET.SubElement(parent_element, field_name).append(ET.Comment(f" Type '{field_info.annotation}' (unresolved ForwardRef) - {description} "))
                     continue
            # Lists
            if origin is list or origin is List:
                list_element = ET.SubElement(parent_element, field_name); list_element.append(ET.Comment(comment_text))
                item_type = args[0] if args else Any
                if item_type is Any: ET.SubElement(list_element, "item").append(ET.Comment(" Generic list item "))
                elif isinstance(item_type, type) and issubclass(item_type, StructuredXML):
                     item_tag = item_type._get_model_root_tag(); list_element.append(ET.Comment(f" Contains <{item_tag}> elements "))
                     item_element = ET.Element(item_tag); cls._build_schema_element(item_element, item_type); list_element.append(item_element)
                else:
                    item_tag = cls._get_list_item_tag(item_type); item_element = ET.SubElement(list_element, item_tag); item_element.append(ET.Comment(f" Example item of type {item_type.__name__} "))
            # Nested Models
            elif isinstance(field_type, type) and issubclass(field_type, StructuredXML):
                nested_element = ET.SubElement(parent_element, field_name); nested_element.append(ET.Comment(comment_text)); cls._build_schema_element(nested_element, field_type)
            # Primitives
            else: ET.SubElement(parent_element, field_name).append(ET.Comment(comment_text))

    # --- XML Parsing (Case-Insensitive) ---

    @classmethod
    def parse_xml(cls: Type[T], xml_string: str) -> T:
        """
        Parses an XML string case-insensitively into an instance of this Pydantic model.
        Args:
            xml_string: The XML string to parse.
        Returns:
            An instance of the model populated with data from the XML.
        Raises:
            ValueError: If the XML is malformed or doesn't match the expected structure (case-insensitive).
            pydantic.ValidationError: If the extracted data fails Pydantic validation.
        """
        try:
            # Basic cleaning (can be adjusted)
            cleaned_xml = re.sub(r'```xml\n?', '', xml_string, flags=re.IGNORECASE)
            cleaned_xml = cleaned_xml.replace('```', '').strip()
            if not cleaned_xml:
                raise ValueError("Input XML string is empty after cleaning")
            root_element = ET.fromstring(cleaned_xml)
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML format: {e}") from e
        except Exception as e:
             raise ValueError(f"Error preparing XML for parsing: {e}") from e

        # --- Case-Insensitive Root Tag Check ---
        expected_root_tag_lower = cls._get_model_root_tag().lower()
        actual_root_tag_lower = root_element.tag.lower()
        if actual_root_tag_lower != expected_root_tag_lower:
            # Report original casing in error for clarity
            raise ValueError(f"Expected root tag like '<{cls._get_model_root_tag()}>' (case-insensitive) but found '<{root_element.tag}>'")
        # --- End Case-Insensitive Check ---

        data = cls._parse_element_to_dict(root_element, cls)
        return cls.model_validate(data)

    @classmethod
    def _parse_element_to_dict(cls, element: ET.Element, model_type: Type['StructuredXML']) -> Dict[str, Any]:
        """Recursively parses XML element case-insensitively into a dict based on model."""
        data: Dict[str, Any] = {}

        # --- Create lookup dict with LOWERCASE keys ---
        children_by_tag_lower: Dict[str, List[ET.Element]] = {}
        for child in element:
            # Ignore comments, processing instructions, etc.
            if isinstance(child.tag, str):
                children_by_tag_lower.setdefault(child.tag.lower(), []).append(child)
        # --- End Lowercase Dict ---

        for field_name, field_info in model_type.model_fields.items():
            field_type = field_info.annotation
            origin = get_origin(field_type)
            args = get_args(field_type)
            is_optional = False

            if origin is Union: # Handle Optional[T]
                union_args = [arg for arg in args if arg is not type(None)]
                if len(union_args) == 1 and type(None) in args:
                    field_type = union_args[0]; origin = get_origin(field_type); args = get_args(field_type); is_optional = True

            if isinstance(field_type, ForwardRef): # Handle ForwardRef
                 try:
                     global_ns, local_ns = model_type.__module__.__dict__, dict(vars(model_type))
                     field_type = field_type._evaluate(global_ns, local_ns or None, set())
                     origin = get_origin(field_type); args = get_args(field_type)
                 except NameError as e:
                      print(f"Warning: Could not resolve ForwardRef '{field_info.annotation}' for field '{field_name}'. Skipping. Error: {e}")
                      continue

            # --- Case-Insensitive Lookup ---
            field_name_lower = field_name.lower()
            child_elements = children_by_tag_lower.get(field_name_lower)
            # --- End Case-Insensitive Lookup ---

            if not child_elements:
                continue # Let Pydantic handle missing fields

            # Handle Lists
            if origin is list or origin is List:
                item_type = args[0] if args else Any
                parsed_list = []
                list_container_element = child_elements[0] # Assume first element is the container

                item_tag = cls._get_list_item_tag(item_type) # Canonical item tag
                item_tag_lower = item_tag.lower() # Lowercase for comparison

                # --- Manual Case-Insensitive Find for List Items ---
                list_item_elements = [
                    child for child in list_container_element
                    if isinstance(child.tag, str) and child.tag.lower() == item_tag_lower
                ]
                # --- End Manual Find ---

                for item_element in list_item_elements:
                    if isinstance(item_type, type) and issubclass(item_type, StructuredXML):
                        # Recursively parse structured item
                        parsed_list.append(cls._parse_element_to_dict(item_element, item_type))
                    else:
                        # Parse primitive item
                        value_str = item_element.text.strip() if item_element.text else ""
                        try: parsed_list.append(cls._parse_primitive(value_str, item_type))
                        except ValueError: parsed_list.append(value_str) # Pass raw string if needed

                data[field_name] = parsed_list

            # Handle Nested StructuredXML Models (non-list)
            elif isinstance(field_type, type) and issubclass(field_type, StructuredXML):
                if len(child_elements) > 1: print(f"Warning: Found multiple <{child_elements[0].tag}> elements for non-list field '{field_name}'. Using first.")
                data[field_name] = cls._parse_element_to_dict(child_elements[0], field_type)

            # Handle Primitive Types (non-list)
            else:
                if len(child_elements) > 1: print(f"Warning: Found multiple <{child_elements[0].tag}> elements for non-list field '{field_name}'. Using first.")
                value_str = child_elements[0].text.strip() if child_elements[0].text else ""
                try: data[field_name] = cls._parse_primitive(value_str, field_type)
                except ValueError: data[field_name] = value_str # Pass raw string

        return data

    @classmethod
    def _parse_primitive(cls, value: str, field_type: Type) -> Any:
        # (Primitive parsing remains unchanged)
        value = value.strip()
        if field_type is type(None): return None if not value else value
        if not value: return value # Return empty string for Pydantic validation
        try:
            if field_type is str: return value
            if field_type is int: return int(value)
            if field_type is float: return float(value)
            if field_type is bool: return value.lower() in ('true', 'yes', '1', 'y')
            return value # Fallback for Pydantic types (date, enum etc)
        except (ValueError, TypeError): return value # Let Pydantic try if direct parse fails


    # --- XML Serialization (Case-Sensitive Output) ---

    def to_xml(self, pretty: bool = False, indent_spaces: int = 4) -> str:
        # (Serialization logic remains unchanged - outputs canonical case)
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
            except Exception as e: print(f"Warning: Failed to pretty-print XML: {e}"); return rough_string
        else: return rough_string

    def _add_fields_to_element(self, parent_element: ET.Element, data: Dict[str, Any], model_type: Type['StructuredXML']):
        # (Serialization building logic remains unchanged - uses canonical case)
        for field_name, field_value in data.items():
            if field_value is None: continue
            field_info = model_type.model_fields.get(field_name);
            if not field_info: continue
            field_type = field_info.annotation; origin = get_origin(field_type); args = get_args(field_type)
            if origin is Union:
                union_args = [arg for arg in args if arg is not type(None)]
                if len(union_args) == 1 and type(None) in args: field_type = union_args[0]; origin = get_origin(field_type); args = get_args(field_type)
            if isinstance(field_type, ForwardRef):
                 try:
                     global_ns, local_ns = model_type.__module__.__dict__, dict(vars(model_type))
                     field_type = field_type._evaluate(global_ns, local_ns or None, set()); origin = get_origin(field_type); args = get_args(field_type)
                 except NameError: print(f"Warning: Could not resolve ForwardRef for serialization: {field_info.annotation}. Skipping field '{field_name}'."); continue
            # Lists
            if isinstance(field_value, list):
                list_element = ET.SubElement(parent_element, field_name); item_type = args[0] if args else Any
                for item in field_value:
                    if item is None: continue
                    item_model_type = item.__class__ if isinstance(item, StructuredXML) else (item_type if isinstance(item_type, type) and issubclass(item_type, StructuredXML) else None)
                    if item_model_type:
                         item_tag = item_model_type._get_model_root_tag(); item_element = ET.SubElement(list_element, item_tag)
                         item_data = item.model_dump(mode='python', by_alias=False) if isinstance(item, StructuredXML) else item; self._add_fields_to_element(item_element, item_data, item_model_type)
                    else: item_tag = self._get_list_item_tag(item_type); item_element = ET.SubElement(list_element, item_tag); item_element.text = str(item)
            # Nested Models
            elif isinstance(field_value, StructuredXML) or (isinstance(field_value, dict) and isinstance(field_type, type) and issubclass(field_type, StructuredXML)):
                 nested_model_type = field_value.__class__ if isinstance(field_value, StructuredXML) else field_type; nested_element = ET.SubElement(parent_element, field_name)
                 nested_data = field_value.model_dump(mode='python', by_alias=False) if isinstance(field_value, StructuredXML) else field_value; self._add_fields_to_element(nested_element, nested_data, nested_model_type)
            # Primitives
            else: element = ET.SubElement(parent_element, field_name); element.text = str(field_value)


# =============================================================================
# Instruction Template Classes (Unchanged)
# =============================================================================
# (SubList, Example, InstructionTemplate classes remain exactly as before)
class SubList(BaseModel):
    header: str
    items: List[str]
class Example(BaseModel):
    input: str; output: str; explanation: Optional[str] = None
class InstructionTemplate(BaseModel):
    task: str = Field(..., description="Main instruction")
    context: Optional[str] = Field(None, description="Background info")
    response_schema: Optional[Union[StructuredXML, Type[StructuredXML], str]] = Field(None, description="Output format")
    guidelines: Optional[List[Union[str, SubList]]] = Field(None, description="Rules")
    examples: Optional[List[Example]] = Field(None, description="Few-shot examples")
    tone: Optional[str] = Field(None, description="Desired tone")
    note: Optional[str] = Field(None, description="Additional notes")
    custom_sections: Optional[Dict[str, str]] = Field(None, description="Custom sections")
    sections_order: Optional[List[str]] = Field(None, description="Custom ordering")
    variables: Optional[Dict[str, Any]] = Field(None, description="Template variables")
    def render(self, **extra_vars) -> str:
        all_vars = {**(self.variables or {}), **extra_vars}
        section_order = self.sections_order or ["task", "context", "response_schema", "guidelines", "examples", "tone", "custom_sections", "note"]
        sections = []
        processed_custom = set() # Keep track of custom sections rendered individually
        for section_name in section_order:
            rendered = None
            if section_name in (self.custom_sections or {}):
                 rendered = self._render_custom_section(section_name, vars=all_vars)
                 processed_custom.add(section_name)
            elif section_name == "custom_sections": # Handle the generic 'custom_sections' keyword
                 rendered = self._render_all_remaining_custom_sections(processed_custom, all_vars)
            else: # Handle built-in sections
                 rendered = self._render_section(section_name, vars=all_vars)
            if rendered: sections.append(rendered)
        return "\n\n".join(sections)

    def _render_section(self, section: str, vars: Dict[str, Any]) -> Optional[str]:
        content = None
        if section == "task" and self.task: content = f"{self._substitute_variables(self.task, vars)}" # No header needed usually
        elif section == "context" and self.context: content = f"# Context:\n{self._substitute_variables(self.context, vars)}"
        elif section == "response_schema" and self.response_schema:
            schema_str = ""; header = "# Response Format:"
            if isinstance(self.response_schema, type) and issubclass(self.response_schema, StructuredXML):
                schema_str = self.response_schema.xml_schema(indent_spaces=2); header = "# Generate your response in the following XML format:"
            elif isinstance(self.response_schema, StructuredXML):
                schema_str = self.response_schema.__class__.xml_schema(indent_spaces=2); header = "# Generate your response in the following XML format:"
            else: schema_str = self._substitute_variables(str(self.response_schema), vars)
            if schema_str.strip().startswith('<') and schema_str.strip().endswith('>'): content = f"{header}\n\n```xml\n{schema_str}```"
            else: content = f"{header}\n{schema_str}"
        elif section == "guidelines" and self.guidelines:
            items = []; idx = 1
            for g in self.guidelines:
                if isinstance(g, str): items.append(f"{idx}. {self._substitute_variables(g, vars)}"); idx += 1
                elif isinstance(g, SubList):
                    sub_items = [f"\n   - {self._substitute_variables(i, vars)}" for i in g.items]
                    items.append(f"{idx}. {self._substitute_variables(g.header, vars)}:{''.join(sub_items)}"); idx += 1
            if items: content = f"# Guidelines:" + "\n".join([""] + items)
        elif section == "examples" and self.examples:
            items = []
            for i, ex in enumerate(self.examples, 1):
                inp = self._substitute_variables(ex.input, vars); outp = self._substitute_variables(ex.output, vars)
                expl = f"\n\n> Explanation: {self._substitute_variables(ex.explanation, vars)}" if ex.explanation else ""
                items.append(f"## Example {i}\nInput:\n```\n{inp}\n```\nOutput:\n```xml\n{outp}\n```{expl}")
            if items: content = "# Examples:\n\n" + "\n\n".join(items)
        elif section == "tone" and self.tone: content = f"# Tone:\n{self._substitute_variables(self.tone, vars)}"
        elif section == "note" and self.note: content = f"# Note:\n{self._substitute_variables(self.note, vars)}"
        return content

    def _render_custom_section(self, title: str, content: Optional[str] = None, vars: Optional[Dict[str, Any]] = None) -> Optional[str]:
         vars = vars or {}; content = content or (self.custom_sections or {}).get(title)
         if not content: return None
         formatted_title = title.replace('_', ' ').title(); content = self._substitute_variables(content, vars)
         return f"# {formatted_title}:\n{content}"

    def _render_all_remaining_custom_sections(self, processed_custom: set, vars: Dict[str, Any]) -> Optional[str]:
        """Renders custom sections not explicitly placed by sections_order."""
        parts = []
        for title, content in (self.custom_sections or {}).items():
             if title not in processed_custom:
                  rendered = self._render_custom_section(title, content, vars)
                  if rendered: parts.append(rendered)
        return "\n\n".join(parts) if parts else None

    def _substitute_variables(self, text: str, vars: Dict[str, Any]) -> str:
        if not text: return ""
        for var_name, value in vars.items():
            pattern = r'\{\{\s*' + re.escape(str(var_name)) + r'\s*\}\}'
            text = re.sub(pattern, str(value), text)
        return text
