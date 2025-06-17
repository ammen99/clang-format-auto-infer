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

def parse_options(html_content):
    """Parses HTML content to extract clang-format options and their values."""
    soup = BeautifulSoup(html_content, 'lxml') # Use lxml parser

    options_data = {}
    options_section_heading = None

    # Find the section containing the options
    # Search for an h2 tag whose text content starts with "Configurable Format Style Options"
    for h2_tag in soup.find_all('h2'):
        if h2_tag.get_text(strip=True).startswith('Configurable Format Style Options'):
            options_section_heading = h2_tag
            break

    if not options_section_heading:
        print("Could not find the 'Configurable Format Style Options' section.", file=sys.stderr)
        return options_data # Return empty if section not found

    # Find the first definition list (<dl>) after the heading
    current_element = options_section_heading.find_next_sibling()
    first_dl = None
    while current_element:
        if current_element.name == 'dl': # type: ignore
            first_dl = current_element
            break
        # Stop if we hit the next major section heading (h2) before finding a dl
        if current_element.name == 'h2': # type: ignore
             break
        current_element = current_element.find_next_sibling()

    if not first_dl:
        print("Could not find the start of the options list (<dl> tag) after the heading.", file=sys.stderr)
        return options_data

    # Iterate through the definition list items (<dl> tags) starting from the first one found
    current_dl = first_dl
    while current_dl:
        # Process the current <dl> tag if it's a definition list
        if current_dl.name == 'dl': # type: ignore
            dt_tag = current_dl.find('dt') # type: ignore
            dd_tag = current_dl.find('dd') # type: ignore

            if dt_tag and dd_tag:
                option_name = None
                option_type = None

                # Extract option name from strong tag within dt
                strong_tag = dt_tag.find('strong') # type: ignore
                if strong_tag:
                    option_name = strong_tag.get_text().strip() # type: ignore

                # Extract option type from code tag within dt, usually right after strong
                if strong_tag:
                    next_sibling = strong_tag.next_sibling # type: ignore
                    while next_sibling:
                        if next_sibling.name == 'code': # type: ignore
                            option_type = next_sibling.get_text().strip()
                            break
                        # Stop if we hit another tag that's not just whitespace/text
                        if hasattr(next_sibling, 'name') and next_sibling.name is not None: # type: ignore
                             break
                        next_sibling = next_sibling.next_sibling


                if option_name and option_type:
                    values = []
                    # Special case for Boolean types
                    if option_type == 'Boolean':
                        values = ['true', 'false']
                    else:
                        # Look for "Possible values:" list in the dd tag
                        # Find the text node "Possible values:"
                        possible_values_heading = dd_tag.find(string=re.compile(r'Possible values:')) # type: ignore
                        if possible_values_heading:
                            # The list is usually immediately after the heading
                            values_list = possible_values_heading.find_next('ul')
                            if values_list:
                                for li in values_list.find_all('li'): # type: ignore
                                    # Extract the configuration value
                                    # It's often in a code tag, sometimes after "(in configuration: "
                                    li_text = li.get_text().strip()
                                    config_value_match = re.search(r'\(in configuration:\s*(.*?)\)', li_text)
                                    if config_value_match:
                                        values.append(config_value_match.group(1).strip())
                                    else:
                                        # Fallback: get the text from the first code tag in the li
                                        code_tag_value = li.find('code') # type: ignore
                                        if code_tag_value:
                                            values.append(code_tag_value.get_text().strip()) # type: ignore
                                        # If no specific format found, maybe skip or add raw text? Let's skip for now.

                    # Add sane values for Integer/Unsigned options related to Offset or Width
                    if option_type in ['Integer', 'Unsigned']:
                        if 'offset' in option_name.lower() or 'width' in option_name.lower():
                            if option_type == 'Integer':
                                sane_int_values = [-4, -2, 0, 1, 2, 3, 4, 8]
                            elif option_type == 'Unsigned':
                                sane_int_values = [0, 1, 2, 3, 4, 8]
                            
                            # Use a set to handle uniqueness, then convert back to a sorted list
                            all_values_set = set(values)
                            for val in sane_int_values:
                                all_values_set.add(str(val))
                            # Sort numerically if possible, otherwise alphabetically
                            values = sorted(list(all_values_set), key=lambda x: int(x) if x.lstrip('-').isdigit() else x)

                    options_data[option_name] = {
                        'type': option_type,
                        'possible_values': values if values else None # Store None if no specific values listed
                    }
                elif option_name:
                     # Handle cases where type might not be found
                     print(f"Warning: Could not parse type for option '{option_name}'", file=sys.stderr)


        # Move to the next sibling element
        current_dl = current_dl.find_next_sibling()
        # Stop if we hit the next major section heading (h2)
        if current_dl and current_dl.name == 'h2': # type: ignore
             break

    return options_data

if __name__ == '__main__':
    url = "https://clang.llvm.org/docs/ClangFormatStyleOptions.html"
    html_content = fetch_html_content(url)

    if html_content:
        options_dict = parse_options(html_content)
        if options_dict:
            # Convert the dictionary format to the desired list format
            options_list = []
            for name, info in options_dict.items():
                options_list.append({
                    'name': name,
                    'type': info['type'],
                    'possible_values': info['possible_values']
                })

            # Output the list as a JSON string
            print(json.dumps(options_list, indent=2))
        else:
            # Print an empty JSON list if no options were found/parsed
            print("[]")
            sys.exit(0) # Exit successfully even if no options were found

    else:
        sys.exit(1) # Exit with an error code if fetching failed
