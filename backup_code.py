# === Apply Media Tier Logic ===
def apply_media_tier_logic(df):
    try:
        file_id = "1LIcEKO-fdXfo1v-IUeU64He5mh6ti1nc"  # ID file Excel update Media Tier

        # Download file Excel
        download_url_media_tier = f"https://drive.google.com/uc?id={file_id}"
        response_media_tier = requests.get(download_url_media_tier)
        xls_media_tier = pd.ExcelFile(BytesIO(response_media_tier.content))

        # Load the relevant sheets
        sheet_client = pd.read_excel(xls_media_tier, sheet_name="Le Minerale - from Client")
        sheet_online = pd.read_excel(xls_media_tier, sheet_name="Online with AVE - Updated")
        
        load_success = True
    except Exception as e:
        st.error(f"❌ Gagal load file Media Tier dari Google Drive: {e}")
        load_success = False

    if load_success:

        # Mapping 'Media Name' to 'Media Tier' for both sheets
        media_tier_dict = {}
        
        # Get media tier from the "Le Minerale - from Client"
        for index, row in sheet_client.iterrows():
            media_name = row['Media Name']
            media_tier = row['Media Tier']
            media_tier_dict[media_name] = media_tier

        # Get media tier from the "Online with AVE - Updated"
        for index, row in sheet_online.iterrows():
            media_name = row['Media Name']
            media_tier = row['Media Tier']
            if media_name not in media_tier_dict:
                media_tier_dict[media_name] = media_tier

        # Apply media tier to the raw data
        for index, row in df[df['Media Name'].notna()].iterrows():
            media_name = row['Media Name']
            
            # Check if Media Name is in our media_tier_dict, otherwise apply Ad Value logic
            if media_name in media_tier_dict:
                df.at[index, 'Media Tier'] = media_tier_dict[media_name]
            else:
                ad_value = row['Ad Value']
                if ad_value >= 18000000:
                    df.at[index, 'Media Tier'] = 1
                elif ad_value >= 12600000:
                    df.at[index, 'Media Tier'] = 2
                else:
                    df.at[index, 'Media Tier'] = 3
        return df
    
    else:
        st.stop()

