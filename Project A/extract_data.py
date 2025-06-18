import pandas as pd

def extract_excel_sheets_to_txt(excel_filename='data.xlsx'):
    """
    Reads an Excel file and saves each sheet as a separate text file.
    
    Args:
        excel_filename (str): The name of the Excel file to process.
    """

    try:
        xls = pd.ExcelFile(excel_filename)
        sheet_names = xls.sheet_names
        
        print(f"Found {len(sheet_names)} sheets in '{excel_filename}': {', '.join(sheet_names)}")
        print("-" * 30)

        for sheet_name in sheet_names:
            print(f"Processing sheet: '{sheet_name}'...")
            
            df = pd.read_excel(xls, sheet_name=sheet_name, header=None)
            
            output_filename = f"{sheet_name}.txt"
            
            df.to_csv(output_filename, sep=',', index=False, header=False, na_rep='')
            
            print(f"  -> Successfully saved data to '{output_filename}'")
            
        print("-" * 30)
        print("Extraction complete. All sheets have been saved as .txt files.")

    except Exception as e:
        print(f"\nAn error occurred during processing: {e}")
        print("Please ensure the file is a valid Excel file and you have the necessary permissions.")

# --- Main execution block ---
if __name__ == "__main__":
    extract_excel_sheets_to_txt()