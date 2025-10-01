# Python 3.13
# tariff_functions.py 20/07/2025 - the file groups together relevant functions to parse tariff columns in the
# "tariff_database". It supports the main file "tariff_algo.ipynb".
# Note: place the two files in the same directory before usage.

import re
import numpy as np
import pandas as pd

# st_duty() standardizes a tariff entry - ad valorem, specific and compound.
# Together with extract_unit, the output consists in a dictionary of the form {'adval': , 'specific': , 'unit': }
def standardize_duty(duty, tariff_unit_to_description):
    duty = sanitize_entry(duty)
    duty = duty.strip().lower()

    if not duty or duty in ['n/a', 'not applicable', 'nan']:
        return {'adval': 0.0, 'specific': [], 'unit': None}
    try:
        if np.isnan(float(duty)):
            return {'adval': 0.0, 'specific': [], 'unit': None}
    except ValueError:
        pass

    if any(keyword in duty for keyword in ['no additional', 'suspended', 'free', 'ensemble', 'garment', 'absence of']):
        return {'adval': 0.0, 'specific': [], 'unit': None}

    if 'less' in duty:  # Handling 'less' cases by returning the first tariff
        duty = duty.split('less')[0].strip()

    if duty == '33 1/3%':  # Particularity of this data set.
        return {'adval': 0.3333, 'specific': [], 'unit': None}

    duties = duty.split('+')
    specific = []
    adval = []

    # COMPOUND $0.10/kg + 5%
    for section in duties:
        section = section.strip()
        if not section:
            continue

        if '%' in section:  # Handling ad valorem component.
            av_comp = float(section.split('%')[0].strip()) / 100
            adval.append(av_comp)

        else:  # Handling specific component.
            unit_found = None
            s_rate = None
            if '/' in section:
                cleared = section.split('/')[0].strip()
            else:
                cleared = section
            if 'cents' in cleared:
                s_rate = float(cleared.strip('cents').strip()) / 100
            elif '¢' in cleared:
                s_rate = float(cleared.strip('¢').strip()) / 100
            elif cleared.startswith('$'):
                s_rate = float(cleared.lstrip('$'))

            if tariff_unit_to_description:
                unit_found = extract_unit(section, tariff_unit_to_description)
            if s_rate is not None:
                specific.append({'rate': s_rate, 'unit': unit_found})
            # Making sure to handle cases where two or more spec components have more than 1 unit.
    if not adval and not specific:
        return {'adval': None, 'specific': [], 'raw': duty}
    return {
        'adval': np.mean(adval) if adval else 0.0,
        'specific': specific,
        'raw': duty
    }


# extract_unit() extracts the units of measure of the tariff entry. It is called inside st_duty().
def extract_unit(text, tariff_unit_to_description):
    text = text.lower()
    # Normalize Unicode symbols.
    text = text.replace('²', '2').replace('³', '3').replace('¢', 'cents')

    for raw_unit in sorted(tariff_unit_to_description.keys(),
                           key=lambda x: -len(x)):  # Ordering keys in descending order (longest first).
        # Regex with word boundaries (\b) to avoid false positives (eg 'g' found inside 'frog')
        if raw_unit in {'g', 't', 'no', 'ct', 'm', 'l'}:
            pattern = rf'\b{re.escape(raw_unit)}\b'
        else:
            pattern = rf'\b{re.escape(raw_unit)}\b'
        if re.search(pattern, text):
            if raw_unit in ['component kilograms', 'kilograms total sugars,']:
                raw_unit = 'kg'
            transformed_unit = tariff_unit_to_description[
                raw_unit]  # e.g. kg is returned as 'kilograms' to ease up the match with amount_units
            return transformed_unit
    return None


# Deals with units conversion if necessary. It uses unit_conversion_map.
def get_converted_quantity(specific_unit, unit1, quantity1, unit2, quantity2, unit_conversion_map):
    # Edge cases
    if unit1 in ['component kilograms', 'kilograms total sugars']:
        unit1 = 'kilograms'
    if unit2 in ['component kilograms', 'kilograms total sugars']:
        unit2 = 'kilograms'
    # Exact match
    if unit1 == specific_unit:
        return quantity1
    elif specific_unit == unit2:
        return quantity2

    # Try conversion from unit1
    if (unit1, specific_unit) in unit_conversion_map:
        return quantity1 * unit_conversion_map[(unit1, specific_unit)]
    # Try conversion from unit2
    if (unit2, specific_unit) in unit_conversion_map:
        return quantity2 * unit_conversion_map[(unit2, specific_unit)]

    # No match or convertible quantity found
    return None


# sanitize_entry() makes sure that 'entry' is a string, as st_duty() works with strings only.
def sanitize_entry(entry):
    if isinstance(entry, list):
        return ' '.join(str(e) for e in entry if e)
    elif pd.isna(entry):
        return ''
    else:
        return str(entry)