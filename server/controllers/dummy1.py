def update_excel_and_save_to_s3(file_details, bucket_name, prefix, qa_list):
    # Initialize S3 client
    s3 = boto3.client('s3')

    # Extract file name and sheet details from the input parameter
    file_name = file_details["file_name"]
    question_sheet_details = file_details["question_sheet_details"]

    # List the files in the S3 bucket
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

    # Filter the files to find the specific Excel file
    excel_files = [item['Key'] for item in response.get('Contents', []) if item['Key'].endswith('.xlsx')]

    file_key = None
    for key in excel_files:
        if file_name in key:
            file_key = key
            break

    if not file_key:
        raise FileNotFoundError(f"File {file_name} not found in S3 bucket {bucket_name} with prefix {prefix}.")

    print(f"Processing file: {file_key}")

    # Download the file to a BytesIO object
    file_stream = BytesIO()
    s3.download_fileobj(bucket_name, file_key, file_stream)

    # Seek to the beginning of the BytesIO object
    file_stream.seek(0)

    # Load the Excel file with openpyxl
    wb = load_workbook(file_stream)

    # Iterate through the provided sheet details
    for sheet_detail in question_sheet_details:
        sheet_name = sheet_detail["sheet_name"]
        question_column = sheet_detail["question_column"]  # Position like "A", "B", etc.
        answer_column = sheet_detail["answer_column"]  # Position like "B", "C", etc.

        # Check if the sheet exists in the workbook
        if sheet_name not in wb.sheetnames:
            print(f"Sheet {sheet_name} not found in the workbook. Skipping.")
            continue

        ws = wb[sheet_name]

        # Convert column letters to indices (1-based, subtract 1 for 0-based indexing)
        question_col_idx = column_index_from_string(question_column)
        answer_col_idx = column_index_from_string(answer_column)

        # Iterate through rows to find the matching question and update the answer
        for row in ws.iter_rows(min_row=2, max_row=ws.max_row):  # Skipping header row
            row_question = row[question_col_idx - 1].value  # 1-based index to 0-based
            for qa_item in qa_list:
                if row_question == qa_item["question"]:
                    # Update the answer in the corresponding answer column
                    ws.cell(row=row[0].row, column=answer_col_idx, value=qa_item["answer"])
                    break

    # Save the updated workbook back to S3
    save_to_s3(wb, bucket_name, file_key)


# -- FUNCTION TO UPDATE ANSWERS TO EXCEL --
def update_excel_and_save_to_s3(file_details, bucket_name, prefix, qa_list):
    # Initialize S3 client
    s3 = boto3.client('s3')

    # Extract file name and sheet details from the input parameter
    file_name = file_details["file_name"]
    question_sheet_details = file_details["question_sheet_details"]

    # List the files in the S3 bucket
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

    # Filter the files to find the specific Excel file
    excel_files = [item['Key'] for item in response.get('Contents', []) if item['Key'].endswith('.xlsx')]

    file_key = None
    for key in excel_files:
        if file_name in key:
            file_key = key
            break

    if not file_key:
        raise FileNotFoundError(f"File {file_name} not found in S3 bucket {bucket_name} with prefix {prefix}.")

    print(f"Processing file: {file_key}")

    # Download the file to a BytesIO object
    file_stream = BytesIO()
    s3.download_fileobj(bucket_name, file_key, file_stream)

    # Seek to the beginning of the BytesIO object
    file_stream.seek(0)

    # Load the Excel file with openpyxl
    wb = load_workbook(file_stream)

    # Iterate through the provided sheet details
    for sheet_detail in question_sheet_details:
        sheet_name = sheet_detail["sheet_name"]
        question_column = sheet_detail["question_column"]  # Position like "A", "B", etc.
        answer_column = sheet_detail["answer_column"]  # Position like "B", "C", etc.

        # Check if the sheet exists in the workbook
        if sheet_name not in wb.sheetnames:
            print(f"Sheet {sheet_name} not found in the workbook. Skipping.")
            continue

        ws = wb[sheet_name]

        # Convert column letters to indices (1-based, subtract 1 for 0-based indexing)
        question_col_idx = column_index_from_string(question_column) - 1
        answer_col_idx = column_index_from_string(answer_column) - 1

        # Load data into pandas dataframe for easier row matching
        df = pd.DataFrame(ws.iter_rows(values_only=True))

        # Update the answer column in the Excel sheet for matching questions
        for qa_item in qa_list:
            question = qa_item["question"]
            answer = qa_item["answer"]

            # Find the row where the question matches
            match_index = df[df[question_col_idx] == question].index
            if not match_index.empty:
                # Write the answer to the corresponding cell in the Excel sheet
                ws.cell(row=match_index[0] + 2, column=answer_col_idx + 1, value=answer)

    # Save the updated workbook back to S3
    save_to_s3(wb, bucket_name, file_key)
