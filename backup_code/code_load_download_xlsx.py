# Download file Excel
file_id = "1qKZcRumDYft3SJ-Cl3qB65gwCRcB1rUZ"  # ID file Excel kamu
download_url = f"https://drive.google.com/uc?id={file_id}"
response = requests.get(download_url)
xls = pd.ExcelFile(BytesIO(response.content))

# Load semua sheet yang dibutuhkan
df_project_list = pd.read_excel(xls, sheet_name="Project List")
df_column_setup = pd.read_excel(xls, sheet_name="Column Setup")
df_rules = pd.read_excel(xls, sheet_name="Rules")
df_column_order = pd.read_excel(xls, sheet_name="Column Order Setup")
# ✅ Validasi keberadaan dan isi kolom penting
if df_column_order is None or df_column_order.empty or "Column Name" not in df_column_order.columns or "Hide" not in df_column_order.columns:
    st.error("❌ Gagal load sheet 'Column Order Setup'. Pastikan sheet tersedia dan kolom 'Column Name' serta 'Hide' ada di dalamnya.")
    st.stop()
df_method_1_keyword = pd.read_excel(xls, sheet_name="Method 1 Keyword")
df_method_selection = pd.read_excel(xls, sheet_name="Method Selection")
df_official_account_setup = pd.read_excel(xls, sheet_name="Official Account Setup")

# Load Last Updated dari NOTES!B2
try:
    df_notes = pd.read_excel(xls, sheet_name="NOTES", header=None)
    last_updated = df_notes.iloc[0, 1]
except:
    last_updated = "Unknown"