from bs4 import BeautifulSoup
import requests
import sys
import re

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

    # Find the section containing the options
    # Search for an h2 tag whose text starts with "Configurable Format Style Options"
    options_section_heading = soup.find('h2', string=re.compile(r'^Configurable Format Style Options'))

    if not options_section_heading:
        print("Could not find the 'Configurable Format Style Options' section.", file=sys.stderr)
        return options_data # Return empty if section not found

    # Iterate through the definition list items (dt/dd pairs) following the heading
    current_element = options_section_heading.find_next_sibling()

    while current_element:
        # Look for <dt> tags which contain the option name and type
        if current_element.name == 'dt':
            dt_tag = current_element
            dd_tag = dt_tag.find_next_sibling('dd')

            if dd_tag:
                option_name = None
                option_type = None

                # Extract option name from strong tag
                strong_tag = dt_tag.find('strong')
                if strong_tag:
                    option_name = strong_tag.get_text().strip()

                # Extract option type from code tag within dt
                code_tag_type = dt_tag.find('code')
                if code_tag_type:
                    option_type = code_tag_type.get_text().strip()

                if option_name and option_type:
                    values = []
                    # Look for "Possible values:" list in the dd tag
                    # Find the text node "Possible values:"
                    possible_values_heading = dd_tag.find(string=re.compile(r'Possible values:'))
                    if possible_values_heading:
                        # The list is usually immediately after the heading
                        values_list = possible_values_heading.find_next('ul')
                        if values_list:
                            for li in values_list.find_all('li'):
                                # Extract the configuration value
                                # It's often in a code tag, sometimes after "(in configuration: "
                                li_text = li.get_text().strip()
                                config_value_match = re.search(r'\(in configuration:\s*(.*?)\)', li_text)
                                if config_value_match:
                                    values.append(config_value_match.group(1).strip())
                                else:
                                    # Fallback: get the text from the first code tag in the li
                                    code_tag_value = li.find('code')
                                    if code_tag_value:
                                         values.append(code_tag_value.get_text().strip())
                                    # If no specific format found, maybe skip or add raw text? Let's skip for now.


                    options_data[option_name] = {
                        'type': option_type,
                        'possible_values': values if values else None # Store None if no specific values listed
                    }

        # Move to the next sibling element
        current_element = current_element.find_next_sibling()
        # Stop if we hit the next major section heading (h2)
        if current_element and current_element.name == 'h2':
             break


    return options_data

if __name__ == '__main__':
    url = "https://clang.llvm.org/docs/ClangFormatStyleOptions.html"
    html_content = fetch_html_content(url)
    # Removed the print(html_content) here as it was likely for debugging

    if html_content:
        options = parse_options(html_content)
        if options:
            print("Clang-Format Style Options and Possible Values:")
            print("----------------------------------------------")
            for name, info in options.items():
                print(f"Option: {name}")
                print(f"  Type: {info['type']}")
                if info['possible_values'] is not None: # Check explicitly for None
                    if info['possible_values']:
                         print(f"  Possible Values: {', '.join(info['possible_values'])}")
                    else:
                         print("  Possible Values: (List is empty or could not be parsed)")
                else:
                    # Indicate if it's a simple type or has nested options
                    # Let's just indicate no explicit list was found.
                    print("  Possible Values: (No explicit list provided)")

                print("-" * 30)
        else:
            print("No options found or parsing failed.")
    else:
        sys.exit(1) # Exit with an error code if fetching failed
