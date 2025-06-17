from bs4 import BeautifulSoup # pyright: ignore
import requests
import sys
import re
import json # Import the json library

def fetch_html_content(url):
    """Fetches HTML content from a given URL."""
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}", file=sys.stderr)
        return None

def _extract_option_details(container_tag, parent_name=None):
    """
    Extracts option details (name, type, possible values) from a given BeautifulSoup tag.
    Handles both top-level options (from dd tags) and nested options (from li tags).
    Recursively finds and extracts nested options.

    Args:
        container_tag (Tag): The BeautifulSoup tag (dd or li) containing the option's details.
        parent_name (str, optional): The name of the parent option, if this is a nested option.

    Returns:
        list: A list of dictionaries, where each dictionary represents an option
              (including itself and any nested options).
    """
    options_found = []
    option_name = None
    option_type = None
    values = []

    # Determine where to find the option name and type
    if container_tag.name == 'dd':
        # For top-level options, name and type are in the sibling dt tag
        dt_tag = container_tag.find_previous_sibling('dt')
        if dt_tag:
            strong_tag = dt_tag.find('strong')
            if strong_tag:
                option_name = strong_tag.get_text().strip()
            
            # Extract option type from code tag within dt, usually right after strong
            if strong_tag:
                next_sibling = strong_tag.next_sibling
                while next_sibling:
                    if next_sibling.name == 'code': # type: ignore
                        option_type = next_sibling.get_text().strip()
                        break
                    if hasattr(next_sibling, 'name') and next_sibling.name is not None: # type: ignore
                         break # Stop if we hit another tag that's not just whitespace/text
                    next_sibling = next_sibling.next_sibling

    elif container_tag.name == 'li':
        # For nested options, name and type are in the first code tag within the li
        code_tag_declaration = container_tag.find('code')
        if code_tag_declaration:
            declaration_text = code_tag_declaration.get_text().strip()
            parts = declaration_text.split(' ', 1) # Split only on the first space
            if len(parts) == 2:
                option_type = parts[0].strip()
                option_name = parts[1].strip()
            else:
                # If it doesn't fit 'type name' pattern, it's likely not a valid nested option declaration.
                # Skip processing this li as an option, as per user request for examples/descriptions.
                return [] # This li is not a valid option, so return empty list

    if not option_name or not option_type:
        # If we couldn't get a name/type, we can't process this option
        # This warning is for cases where the dt/dd or li structure is fundamentally broken for an option.
        if container_tag.name == 'dd':
            print(f"Warning: Could not parse top-level option name/type from dt/dd pair.", file=sys.stderr)
        # No warning for li here, as handled above by returning []
        return [] # Return empty list if parsing failed for this item

    # Construct the full option name (e.g., "BraceWrapping.AfterCaseLabel")
    full_option_name = f"{parent_name}.{option_name}" if parent_name else option_name

    # --- Extract possible values for the current option ---
    if option_type == 'Boolean':
        values = ['true', 'false']
    else:
        # Find the ul that contains nested options, if any (only relevant for dd tags)
        nested_options_ul = None
        if container_tag.name == 'dd':
            nested_heading = container_tag.find('p', string=re.compile(r'Nested configuration flags:'))
            if nested_heading:
                nested_options_ul = nested_heading.find_next_sibling('ul')

        # Look for lists of values. These are typically ul tags.
        # We need to distinguish them from the ul containing nested options.
        for ul_tag in container_tag.find_all('ul', recursive=False): # Only direct ul children
            if ul_tag == nested_options_ul:
                continue # This ul contains nested options, not values for the current option

            # Heuristic to identify a list of values:
            # Check if the ul contains li elements, and those li elements contain a code tag
            # whose text is a single word or matches the "(in configuration: ...)" pattern.
            potential_values_found = False
            current_ul_values = []
            for li in ul_tag.find_all('li', recursive=False):
                li_text = li.get_text().strip()
                config_value_match = re.search(r'\(in configuration:\s*(.*?)\)', li_text)
                if config_value_match:
                    current_ul_values.append(config_value_match.group(1).strip())
                    potential_values_found = True
                else:
                    code_tag_value = li.find('code', recursive=False)
                    if code_tag_value:
                        code_text = code_tag_value.get_text().strip()
                        # If it's a single word or looks like a value, add it.
                        # Avoid adding things that look like 'type name' declarations for nested options.
                        if ' ' not in code_text or not re.match(r'^\w+\s+\w+$', code_text):
                            current_ul_values.append(code_text)
                            potential_values_found = True
                    # else: if no code tag, it's probably not a value list

            if potential_values_found:
                values.extend(current_ul_values)
                # Assuming there's only one primary list of values for an option
                break

    # Add sane values for Integer/Unsigned options related to Offset or Width
    if option_type in ['Integer', 'Unsigned']:
        if 'offset' in full_option_name.lower() or 'width' in full_option_name.lower():
            sane_int_values = []
            if option_type == 'Integer':
                sane_int_values = [-4, -2, 0, 1, 2, 3, 4, 8]
            elif option_type == 'Unsigned':
                sane_int_values = [0, 1, 2, 3, 4, 8]

            all_values_set = set(values)
            for val in sane_int_values:
                all_values_set.add(str(val))
            values = sorted(list(all_values_set), key=lambda x: int(x) if x.lstrip('-').isdigit() else x)

    # Add the current option to the list of found options
    options_found.append({
        'name': full_option_name,
        'type': option_type,
        'possible_values': values if values else None
    })

    # --- Check for nested options within the current container_tag (only for dd tags) ---
    if container_tag.name == 'dd':
        nested_heading = container_tag.find('p', string=re.compile(r'Nested configuration flags:'))
        if nested_heading:
            nested_ul = nested_heading.find_next_sibling('ul')
            if nested_ul:
                # Only direct children li elements, not li from deeper nested lists (like "Possible values")
                for li_tag in nested_ul.find_all('li', recursive=False): 
                    # Recursively call to extract details for nested options
                    nested_options = _extract_option_details(li_tag, full_option_name)
                    options_found.extend(nested_options)

    return options_found


def parse_options(html_content):
    """Parses HTML content to extract clang-format options and their values."""
    soup = BeautifulSoup(html_content, 'lxml') # Use lxml parser

    all_options_data = {} # Use a dict to ensure unique names and easy lookup

    options_section_heading = None
    for h2_tag in soup.find_all('h2'):
        if h2_tag.get_text(strip=True).startswith('Configurable Format Style Options'):
            options_section_heading = h2_tag
            break

    if not options_section_heading:
        print("Could not find the 'Configurable Format Style Options' section.", file=sys.stderr)
        return all_options_data

    current_element = options_section_heading.find_next_sibling()
    first_dl = None
    while current_element:
        if current_element.name == 'dl': # type: ignore
            first_dl = current_element
            break
        if current_element.name == 'h2': # type: ignore
             break
        current_element = current_element.find_next_sibling()

    if not first_dl:
        print("Could not find the start of the options list (<dl> tag) after the heading.", file=sys.stderr)
        return all_options_data

    current_dl = first_dl
    while current_dl:
        if current_dl.name == 'dl': # type: ignore
            dd_tag = current_dl.find('dd') # type: ignore
            if dd_tag:
                # Call the helper function to extract details for this top-level option and its nested ones
                extracted_options = _extract_option_details(dd_tag)
                for option_info in extracted_options:
                    all_options_data[option_info['name']] = {
                        'type': option_info['type'],
                        'possible_values': option_info['possible_values']
                    }

        current_dl = current_dl.find_next_sibling()
        if current_dl and current_dl.name == 'h2': # type: ignore
             break

    return all_options_data

if __name__ == '__main__':
    url = "https://clang.llvm.org/docs/ClangFormatStyleOptions.html"
    html_content = fetch_html_content(url)

    if html_content:
        options_dict = parse_options(html_content)
        if options_dict:
            options_list = []
            # Sort options by name for consistent output
            for name in sorted(options_dict.keys()):
                info = options_dict[name]
                options_list.append({
                    'name': name,
                    'type': info['type'],
                    'possible_values': info['possible_values']
                })

            print(json.dumps(options_list, indent=2))
        else:
            print("[]")
            sys.exit(0)

    else:
        sys.exit(1)
