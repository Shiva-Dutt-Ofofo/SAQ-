import pandas as pd

def find_matching_columns(df, keywords):
    """Find columns that contain any of the specified keywords."""
    matching_columns = []
    for col in df.columns:
        if any(keyword.lower() in col.lower() for keyword in keywords):
            matching_columns.append(col)
    return matching_columns

def find_next_empty_cell(df, row_idx):
    """Find the index of the next empty cell in a given row."""
    for col_idx in range(len(df.columns)):
        if pd.isna(df.iloc[row_idx, col_idx]) or df.iloc[row_idx, col_idx] == "":
            return col_idx
    return None  # Return None if no empty cell is found

def update_row_with_string(df, row_idx, col_idx, string):
    """Write the string to the specified cell."""
    df.iat[row_idx, col_idx] = string

def update_answer_column(df, query_column, strings_list):
    """Update the next empty cell in a row based on matches in the query column."""
    for idx, query in enumerate(df[query_column]):
        if isinstance(query, str):  # Check if the query is a string
            for string in strings_list:
                if query.strip().lower() in string.lower():
                    next_empty_col_idx = find_next_empty_cell(df, idx)
                    if next_empty_col_idx is not None:
                        update_row_with_string(df, idx, next_empty_col_idx, string)
                    break  # Move to the next row after finding a match

def process_excel_file(file_path, strings_list):
    # Define the keywords to search for in the columns
    keywords = ["query", "questions", "checklist"]

    # Load the Excel file
    excel_file = pd.ExcelFile(file_path)

    # Iterate through each sheet
    for sheet_name in excel_file.sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        
        # Identify columns that contain the keywords
        matching_columns = find_matching_columns(df, keywords)

        # Update the next empty cell in the row for each matching column
        for column in matching_columns:
            update_answer_column(df, column, strings_list)

        # Save the updated sheet back to the Excel file
        with pd.ExcelWriter(file_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)

# Example usage
strings_list = [
    """Question: 1. Describe your application’s architecture and different tiers
2. Explain the design and development phases of your application

Assistant: Answer:

1. The Sharpsell application architecture consists of end customers, services, and data management layers. The end customers interact with the application through microservices, which include features like Learning Enablement. External APIs are used for data exchange between the Sharpsell platform and external systems.

2. During the design and development phases, the system is designed to meet functional requirements, mitigate risks, and ensure security. The design phase involves identifying potential risks, performing a security risk assessment, and developing a conversion plan. The development phase includes creating detailed documentation, testing, and implementing security vulnerability scans and penetration tests. The operations and maintenance phase involves ongoing system monitoring, modifications, and security vulnerability management.""",
    """Question: 4. Do you perform web application vulnerability testing? What is the frequency?

Assistant: Answer: Yes, we perform web application vulnerability testing. The frequency is quarterly.
"""
]

file_path = 'Copy of Cloud Checklist_with CSA mapping_v2.4_SaaS (2).xlsx'
process_excel_file(file_path, strings_list)